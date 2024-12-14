import json
import os
import logging
from typing import Dict, Any
from dotenv import set_key, load_dotenv
from requests_oauthlib import OAuth1Session
from src.connections.base_connection import BaseConnection
from src.helpers import print_h_bar
import tweepy

# Configure module logger
logger = logging.getLogger("connections.twitter_connection")
logger.setLevel(logging.DEBUG)


class TwitterConnectionError(Exception):
    """Base exception for Twitter connection errors"""
    pass


class TwitterConfigurationError(TwitterConnectionError):
    """Raised when there are configuration/credential issues"""
    pass


class TwitterAPIError(TwitterConnectionError):
    """Raised when Twitter API requests fail"""
    pass


class TwitterConnection(BaseConnection):

    def __init__(self):
        super().__init__()
        self._oauth_session = None
        self.actions = {
            "get-latest-tweets": {
                "func": self.get_latest_tweets,
                "args": {
                    "username": "str",
                    "count": "int"
                }
            },
            "post-tweet": {
                "func": self.post_tweet,
                "args": {
                    "message": "str"
                },
            },
            "read-timeline": {
                "func": self.read_timeline,
                "args": {
                    "count": "int"
                },
            },
            "like-tweet": {
                "func": self.like_tweet,
                "args": {
                    "tweet_id": "str"
                },
            },
            "reply-to-tweet": {
                "func": self.reply_to_tweet,
                "args": {
                    "tweet_id": "str",
                    "message": "str"
                },
            }
        }
        logger.debug("TwitterConnection initialized")

    def _make_request(self, method: str, endpoint: str, **kwargs) -> dict:
        """
        Make a request to the Twitter API with error handling

        Args:
            method: HTTP method ('get', 'post', etc.)
            endpoint: API endpoint path
            **kwargs: Additional request parameters

        Returns:
            Dict containing the API response
        """
        logger.debug(f"Making {method.upper()} request to {endpoint}")
        try:
            oauth = self._get_oauth()
            full_url = f"https://api.twitter.com/2/{endpoint.lstrip('/')}"

            response = getattr(oauth, method.lower())(full_url, **kwargs)
            expected_status = 201 if method.lower() == 'post' else 200

            if response.status_code != expected_status:
                logger.error(
                    f"Request failed: {response.status_code} - {response.text}"
                )
                raise TwitterAPIError(
                    f"Request failed with status {response.status_code}: {response.text}"
                )

            logger.debug(f"Request successful: {response.status_code}")
            return response.json()

        except Exception as e:
            logger.error(f"API request failed: {str(e)}")
            if isinstance(e, TwitterAPIError):
                raise
            raise TwitterAPIError(f"API request failed: {str(e)}")

    def _get_credentials(self) -> Dict[str, str]:
        """Get Twitter credentials from environment with validation"""
        logger.debug("Retrieving Twitter credentials")
        load_dotenv()

        required_vars = {
            'TWITTER_CONSUMER_KEY': 'consumer key',
            'TWITTER_CONSUMER_SECRET': 'consumer secret',
            'TWITTER_ACCESS_TOKEN': 'access token',
            'TWITTER_ACCESS_TOKEN_SECRET': 'access token secret',
            'TWITTER_USER_ID': 'user ID'
        }

        credentials = {}
        missing = []

        for env_var, description in required_vars.items():
            value = os.getenv(env_var)
            if not value:
                missing.append(description)
            credentials[env_var] = value

        if missing:
            error_msg = f"Missing Twitter credentials: {', '.join(missing)}"
            logger.error(error_msg)
            raise TwitterConfigurationError(error_msg)

        logger.debug("All required credentials found")
        return credentials

    def _get_oauth(self) -> OAuth1Session:
        """Get or create OAuth session using stored credentials"""
        if self._oauth_session is None:
            logger.debug("Creating new OAuth session")
            try:
                credentials = self._get_credentials()
                self._oauth_session = OAuth1Session(
                    credentials['TWITTER_CONSUMER_KEY'],
                    client_secret=credentials['TWITTER_CONSUMER_SECRET'],
                    resource_owner_key=credentials['TWITTER_ACCESS_TOKEN'],
                    resource_owner_secret=credentials[
                        'TWITTER_ACCESS_TOKEN_SECRET'],
                )
                logger.debug("OAuth session created successfully")
            except Exception as e:
                logger.error(f"Failed to create OAuth session: {str(e)}")
                raise

        return self._oauth_session

    def _get_authenticated_user_id(self) -> str:
        """Get the authenticated user's ID using the users/me endpoint"""
        logger.debug("Getting authenticated user ID")
        try:
            response = self._make_request('get',
                                          'users/me',
                                          params={'user.fields': 'id'})
            user_id = response['data']['id']
            logger.debug(f"Retrieved user ID: {user_id}")
            return user_id
        except Exception as e:
            logger.error(f"Failed to get authenticated user ID: {str(e)}")
            raise TwitterConfigurationError(
                "Could not retrieve user ID") from e

    def _validate_tweet_text(self, text: str, context: str = "Tweet") -> None:
        """Validate tweet text meets Twitter requirements"""
        if not text:
            error_msg = f"{context} text cannot be empty"
            logger.error(error_msg)
            raise ValueError(error_msg)
        if len(text) > 280:
            error_msg = f"{context} exceeds 280 character limit"
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.debug(f"Tweet text validation passed for {context.lower()}")

    def perform_action(self, action_name: str, **kwargs) -> Any:
        """Implementation of abstract method from BaseConnection"""
        logger.debug(f"Performing action: {action_name}")
        if action_name in self.actions:
            return self.actions[action_name]["func"](**kwargs)
        error_msg = f"Unknown action: {action_name}"
        logger.error(error_msg)
        raise TwitterConnectionError(error_msg)

    def configure(self) -> None:
        """Sets up Twitter API authentication"""
        logger.info("Starting Twitter authentication setup")

        # Check existing configuration
        if self.is_configured(verbose=False):
            logger.info("Twitter API is already configured")
            response = input("Do you want to reconfigure? (y/n): ")
            if response.lower() != 'y':
                return

        setup_instructions = [
            "\n🐦 TWITTER AUTHENTICATION SETUP",
            "\n📝 To get your Twitter API credentials:",
            "1. Go to https://developer.twitter.com/en/portal/dashboard",
            "2. Create a new project and app if you haven't already",
            "3. In your app settings, enable OAuth 1.0a with read and write permissions",
            "4. Get your API Key (consumer key) and API Key Secret (consumer secret)"
        ]
        logger.info("\n".join(setup_instructions))
        print_h_bar()

        try:
            # Get account details
            logger.info("\nPlease enter your Twitter API credentials:")
            credentials = {
                'consumer_key':
                input("Enter your API Key (consumer key): "),
                'consumer_secret':
                input("Enter your API Key Secret (consumer secret): ")
            }

            logger.info("Starting OAuth authentication process...")

            # Initialize OAuth flow
            request_token_url = "https://api.twitter.com/oauth/request_token?oauth_callback=oob&x_auth_access_type=write"
            oauth = OAuth1Session(credentials['consumer_key'],
                                  client_secret=credentials['consumer_secret'])

            try:
                fetch_response = oauth.fetch_request_token(request_token_url)
            except ValueError as e:
                logger.error("Failed to fetch request token")
                raise TwitterConfigurationError(
                    "Invalid consumer key or secret") from e

            # Get authorization
            base_authorization_url = "https://api.twitter.com/oauth/authorize"
            authorization_url = oauth.authorization_url(base_authorization_url)

            auth_instructions = [
                "\n1. Please visit this URL to authorize the application:",
                authorization_url,
                "\n2. After authorizing, Twitter will give you a PIN code."
            ]
            logger.info("\n".join(auth_instructions))

            verifier = input("3. Please enter the PIN code here: ")

            # Get access token
            access_token_url = "https://api.twitter.com/oauth/access_token"
            oauth = OAuth1Session(
                credentials['consumer_key'],
                client_secret=credentials['consumer_secret'],
                resource_owner_key=fetch_response.get('oauth_token'),
                resource_owner_secret=fetch_response.get('oauth_token_secret'),
                verifier=verifier)

            oauth_tokens = oauth.fetch_access_token(access_token_url)

            # Save credentials
            if not os.path.exists('.env'):
                logger.debug("Creating new .env file")
                with open('.env', 'w') as f:
                    f.write('')

            # Create temporary OAuth session to get user ID
            temp_oauth = OAuth1Session(
                credentials['consumer_key'],
                client_secret=credentials['consumer_secret'],
                resource_owner_key=oauth_tokens.get('oauth_token'),
                resource_owner_secret=oauth_tokens.get('oauth_token_secret'))

            self._oauth_session = temp_oauth
            user_id = self._get_authenticated_user_id()

            # Save to .env
            env_vars = {
                'TWITTER_USER_ID':
                user_id,
                'TWITTER_CONSUMER_KEY':
                credentials['consumer_key'],
                'TWITTER_CONSUMER_SECRET':
                credentials['consumer_secret'],
                'TWITTER_ACCESS_TOKEN':
                oauth_tokens.get('oauth_token'),
                'TWITTER_ACCESS_TOKEN_SECRET':
                oauth_tokens.get('oauth_token_secret')
            }

            for key, value in env_vars.items():
                set_key('.env', key, value)
                logger.debug(f"Saved {key} to .env")

            logger.info("\n✅ Twitter authentication successfully set up!")
            logger.info(
                "Your API keys, secrets, and user ID have been stored in the .env file."
            )

        except Exception as e:
            error_msg = f"Setup failed: {str(e)}"
            logger.error(error_msg)
            raise TwitterConfigurationError(error_msg)

    def is_configured(self, verbose=True) -> bool:
        """Check if Twitter credentials are configured and valid"""
        logger.debug("Checking Twitter configuration status")
        try:
            credentials = self._get_credentials()

            # Initialize client and validate credentials
            client = tweepy.Client(
                consumer_key=credentials['TWITTER_CONSUMER_KEY'],
                consumer_secret=credentials['TWITTER_CONSUMER_SECRET'],
                access_token=credentials['TWITTER_ACCESS_TOKEN'],
                access_token_secret=credentials['TWITTER_ACCESS_TOKEN_SECRET'])

            client.get_me()
            logger.debug("Twitter configuration is valid")
            return True

        except Exception as e:
            if verbose:
                error_msg = str(e)
                if isinstance(e, TwitterConfigurationError):
                    error_msg = f"Configuration error: {error_msg}"
                elif isinstance(e, TwitterAPIError):
                    error_msg = f"API validation error: {error_msg}"
                logger.error(f"Configuration validation failed: {error_msg}")
            return False

    def read_timeline(self, count=10, **kwargs) -> list:
        """Read tweets from the user's timeline"""
        logger.debug(f"Reading timeline, count: {count}")
        credentials = self._get_credentials()

        params = {
            "tweet.fields": "created_at,author_id,attachments",
            "expansions": "author_id",
            "user.fields": "name,username",
            "max_results": count
        }

        response = self._make_request(
            'get',
            f"users/{credentials['TWITTER_USER_ID']}/timelines/reverse_chronological",
            params=params)

        tweets = response.get("data", [])
        user_info = response.get("includes", {}).get("users", [])

        # Process user information
        user_dict = {
            user['id']: {
                'name': user['name'],
                'username': user['username']
            }
            for user in user_info
        }

        # Enrich tweets with user information
        for tweet in tweets:
            author_id = tweet['author_id']
            author_info = user_dict.get(author_id, {
                'name': "Unknown",
                'username': "Unknown"
            })
            tweet.update({
                'author_name': author_info['name'],
                'author_username': author_info['username']
            })

        logger.debug(f"Retrieved {len(tweets)} tweets")
        return tweets

    def get_latest_tweets(self,
                          username: str,
                          count: int = 10,
                          **kwargs) -> list:
        """Get latest tweets for a user"""
        logger.debug(f"Getting latest tweets for {username}, count: {count}")

        credentials = self._get_credentials()
        params = {
            "tweet.fields": "created_at,public_metrics,text",
            "max_results": min(count, 100),
            "exclude": "retweets,replies"
        }

        response = self._make_request('get',
                                      f"users/{credentials['TWITTER_USER_ID']}/tweets",
                                      params=params)

        tweets = response.get("data", [])
        logger.debug(f"Retrieved {len(tweets)} tweets")
        return tweets

    def post_tweet(self, message: str, **kwargs) -> dict:
        """Post a new tweet"""
        logger.debug("Posting new tweet")
        self._validate_tweet_text(message)

        response = self._make_request('post', 'tweets', json={'text': message})

        logger.info("Tweet posted successfully")
        return response

    def reply_to_tweet(self, tweet_id: str, message: str, **kwargs) -> dict:
        """Reply to an existing tweet"""
        logger.debug(f"Replying to tweet {tweet_id}")
        self._validate_tweet_text(message, "Reply")

        response = self._make_request('post',
                                      'tweets',
                                      json={
                                          'text': message,
                                          'reply': {
                                              'in_reply_to_tweet_id': tweet_id
                                          }
                                      })

        logger.info("Reply posted successfully")
        return response

    def like_tweet(self, tweet_id: str, **kwargs) -> dict:
        """Like a tweet"""
        logger.debug(f"Liking tweet {tweet_id}")
        credentials = self._get_credentials()

        response = self._make_request(
            'post',
            f"users/{credentials['TWITTER_USER_ID']}/likes",
            json={'tweet_id': tweet_id})

        logger.info("Tweet liked successfully")
        return response
