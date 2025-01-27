import json
import time
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from src.connections.postgres import DatabaseHandler
from src.connection_manager import ConnectionManager
from src.helpers import print_h_bar

REQUIRED_FIELDS = ["name", "bio", "traits", "examples", "loop_delay", "config"]

logger = logging.getLogger("agent")

class DiscordAgent:
    def __init__(
            self,
            agent_name: str
    ):
        try:
            agent_path = Path("agents") / f"{agent_name}.json"
            agent_dict = json.load(open(agent_path, "r"))

            missing_fields = [field for field in REQUIRED_FIELDS if field not in agent_dict]
            if missing_fields:
                raise KeyError(f"Missing required fields: {', '.join(missing_fields)}")

            load_dotenv()
            
            # Initialize basic attributes
            self.name = agent_dict["name"]
            self.bio = agent_dict["bio"]
            self.channel_id = os.getenv("CHANNEL_ID")  # Your specific channel ID
            self.loop_delay = agent_dict["loop_delay"]
            self.message_read_count = 10  # Number of messages to read each time
            self.is_llm_set = False

            # Set up database handler
            self.db = DatabaseHandler()

            # Initialize connection manager
            self.connection_manager = ConnectionManager(agent_dict["config"])

            # Validate Discord connection
            if not self.connection_manager.connections.get("discord"):
                raise ValueError("Discord connection not found in configuration")

            if not self.connection_manager.connections["discord"].is_configured():
                logger.warning("Discord connection is not configured. Running configure...")
                if not self.connection_manager.configure_connection("discord"):
                    raise ValueError("Failed to configure Discord connection")

            # Initialize last processed message ID from current latest message
            self.last_processed_message_id = self._get_latest_message_id()
            
            # Add a fallback for `last_processed_message_id`
            if not self.last_processed_message_id:
                self.last_processed_message_id = "0"  # Default value for message ID
                logger.info("No previous messages found. Starting with default last_processed_message_id: '0'")

            logger.info(f"Initialized with last message ID: {self.last_processed_message_id}")

            # Set up LLM provider
            self._setup_llm_provider()

        except Exception as e:
            logger.error("Could not load Discord agent")
            raise e

    def _get_latest_message_id(self) -> str:
        """Get the ID of the latest message in the channel"""
        try:
            messages = self.read_messages()
            print(messages)
            if messages and len(messages) > 0:
                latest_id = messages[0]['id']
                logger.info(f"Got latest message ID: {latest_id}")
                return latest_id
            logger.warning("No messages found to initialize from")
            return None
        except Exception as e:
            logger.error(f"Failed to get latest message ID: {e}")
            return None

    def _setup_llm_provider(self):
        """Setup LLM provider for message generation"""
        llm_providers = self.connection_manager.get_model_providers()
        if not llm_providers:
            raise ValueError("No configured LLM provider found")
        self.model_provider = llm_providers[0]
        self.is_llm_set = True

    def _generate_reply(self, original_message: str) -> str:
        """Generate a reply to a specific message"""
        prompt = f"Generate a reply to this message: '{original_message}'. You are {self.name}, an agent with technical expertise, so your responses may include code snippets or technical explanations based on the query and your context. Discord chats use markdown, so you can take advantage of creating rich text responses that include code blocks, lists, and more. Due to limitations in the Discord response length, your <think></think> logs together with your response will need to be 2000 or fewer in length."
        system_prompt = "\n".join(self.bio)
        similar_content = self.db.get_similar_content(original_message)
        if similar_content:
            context = "\nRelevant context:\n"
            for content in similar_content:
                context += f"- {content}\n"
            # append similar content to the system prompt with an indicator that this is relevant context
            system_prompt += '\n' + 'This is relevant context:\n' + context
        
        return self.connection_manager.perform_action(
            connection_name=self.model_provider,
            action_name="generate-text",
            params=[prompt, system_prompt]
        )

    def read_messages(self) -> list:
        """Read recent messages from the channel"""
        try:
            messages = self.connection_manager.perform_action(
                connection_name="discord",
                action_name="read-messages",
                params=[self.channel_id, self.message_read_count]
            )
            return messages if messages else []
        except Exception as e:
            logger.error(f"Failed to read messages: {e}")
            return []

    def reply_to_message(self, message_id: str, original_message: str) -> bool:
        """Reply to a specific message"""
        try:
            reply = self._generate_reply(original_message)
            
            result = self.connection_manager.perform_action(
                connection_name="discord",
                action_name="reply-to-message",
                params=[self.channel_id, message_id, reply]
            )
            
            if result:
                logger.info(f"Successfully replied to message: '{original_message}' with: '{reply}'")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to reply to message: {e}")
            return False

    def process_new_messages(self):
        """Check for and reply to new messages"""
        if not self.last_processed_message_id:
            logger.warning("No baseline message ID set, skipping message processing")
            return
            
        messages = self.read_messages()
        if not messages:
            return
        
        # Process messages in chronological order (oldest first)
        new_messages = []
        for msg in reversed(messages):
            msg_id = msg['id']
            msg_content = msg.get('message', '')
            author = msg.get('author', '')
            
            # Skip if it's any kind of bot message or reference:
            # 1. From the bot (APP in author)
            # 2. Message references/mentions the bot (@zerecall)
            # 3. Message is a reply to a bot message
            if ('APP' in author or 
                'zerecall' in author.lower() or
                '@zerecall' in msg_content.lower() or
                any(ref in msg_content.lower() for ref in ['@zerecall', '<@zerecall'])):
                logger.debug(f"Skipping bot-related message: {msg_id}")
                continue
            
            # Only process if it's a new message we haven't replied to
            if (msg_id > self.last_processed_message_id):
                new_messages.append(msg)
                logger.info(f"Found new message to process: {msg_id}")
        
        for message in new_messages:
            msg_id = message['id']
            msg_content = message.get('message', '')
            
            # Check if the message has already been replied to using the database
            if not self.db.get_replied_messages(self.channel_id, msg_id):
                if self.reply_to_message(msg_id, msg_content):
                    # Add the message to the database as replied
                    self.db.add_replied_message(self.channel_id, msg_id)
                    self.last_processed_message_id = msg_id
                    logger.info(f"Successfully processed message {msg_id}")
                else:
                    logger.warning(f"Failed to process message {msg_id}")


    def loop(self):
        """Main loop that only replies to new messages"""
        logger.info("\n🤖 Starting Discord reply agent loop...")
        logger.info("Listening for new messages. Press Ctrl+C to stop.")
        print_h_bar()

        try:
            while True:
                try:
                    # Check for and reply to new messages
                    self.process_new_messages()
                    
                    # Quick check interval (1 second) but don't spam the logs
                    time.sleep(self.loop_delay)

                except Exception as e:
                    logger.error(f"\n❌ Error in loop iteration: {e}")
                    time.sleep(2)  # Short delay on error before retry

        except KeyboardInterrupt:
            logger.info("\n🛑 Discord agent loop stopped by user.")
            return