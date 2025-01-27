import os
import logging
import psycopg2
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from psycopg2.extras import Json
from typing import Set, Optional

logger = logging.getLogger("db_handler")

class DatabaseHandler:
    def __init__(self):
        load_dotenv()
            
        self.conn_string = os.getenv("POSTGRES_URL")
        if not self.conn_string:
            raise ValueError("POSTGRES_URL environment variable not set")
        
        self.setup_database()

    def setup_database(self):
        """Initialize database tables"""
        with psycopg2.connect(self.conn_string) as conn:
            with conn.cursor() as cur:
                # Table for message tracking
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS message_tracking (
                        channel_id TEXT NOT NULL,
                        message_id TEXT NOT NULL,
                        processed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (channel_id, message_id)
                    )
                """)
            
                
                conn.commit()

    def get_replied_messages(self, channel_id: str, message_id: str) -> bool:
        """Check if a message has been replied to"""
        with psycopg2.connect(self.conn_string) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT EXISTS(SELECT 1 FROM message_tracking WHERE channel_id = %s AND message_id = %s)",
                    (channel_id, message_id)
                )
                return cur.fetchone()[0]

    def add_replied_message(self, channel_id: str, message_id: str):
        """Record a replied message"""
        with psycopg2.connect(self.conn_string) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO message_tracking (channel_id, message_id) VALUES (%s, %s)",
                    (channel_id, message_id)
                )
                conn.commit()

    def get_similar_content(self, query: str, limit: int = 5) -> list:
        """Get similar content using vector similarity"""
        try:
            load_dotenv()
            openai_api_key=os.getenv("OPENAI_KEY")
            embeddings = OpenAIEmbeddings(
                api_key=openai_api_key
            )
            
            # Generate embedding for query
            query_embedding = embeddings.embed_query(query)
            
            with psycopg2.connect(self.conn_string) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT content, embedding <=> %s::vector AS similarity
                        FROM code_embeddings
                        ORDER BY similarity ASC
                        LIMIT %s
                        """,
                        (f"[{','.join(map(str, query_embedding))}]", limit)
                    )
                    return cur.fetchall()
                    
        except Exception as e:
            logger.error(f"Error getting similar content: {e}")
            return []

    def cleanup_old_messages(self, days: int = 30):
        """Clean up old message records"""
        with psycopg2.connect(self.conn_string) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM message_tracking 
                    WHERE processed_at < NOW() - INTERVAL '%s days'
                    """,
                    (days,)
                )
                conn.commit()