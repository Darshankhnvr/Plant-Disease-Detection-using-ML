#!/usr/bin/env python3
"""
Database connection manager to prevent SQLite locking issues
"""

import sqlite3
import threading
import time
from contextlib import contextmanager

class DatabaseManager:
    """
    Thread-safe database connection manager for SQLite
    """
    
    def __init__(self, db_path='disease_tracking.db'):
        self.db_path = db_path
        self.lock = threading.Lock()
        
    @contextmanager
    def get_connection(self, timeout=30.0):
        """
        Context manager for database connections with proper locking
        """
        conn = None
        try:
            with self.lock:
                conn = sqlite3.connect(
                    self.db_path, 
                    timeout=timeout,
                    check_same_thread=False
                )
                # Enable WAL mode for better concurrency
                conn.execute('PRAGMA journal_mode=WAL')
                conn.execute('PRAGMA synchronous=NORMAL')
                conn.execute('PRAGMA cache_size=10000')
                conn.execute('PRAGMA temp_store=MEMORY')
                
            yield conn
            
        except sqlite3.OperationalError as e:
            if conn:
                conn.rollback()
            if "database is locked" in str(e).lower():
                # Retry once after a short delay
                time.sleep(0.1)
                try:
                    with self.lock:
                        conn = sqlite3.connect(self.db_path, timeout=timeout)
                    yield conn
                except Exception as retry_e:
                    raise Exception(f"Database locked error (retry failed): {str(retry_e)}")
            else:
                raise Exception(f"Database error: {str(e)}")
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()

# Global database manager instance
db_manager = DatabaseManager()

def execute_query(query, params=None, fetch=False, timeout=30.0):
    """
    Execute a query with proper connection management
    """
    with db_manager.get_connection(timeout=timeout) as conn:
        cursor = conn.cursor()
        
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        if fetch:
            if fetch == 'one':
                result = cursor.fetchone()
            elif fetch == 'all':
                result = cursor.fetchall()
            else:
                result = cursor.fetchall()
        else:
            result = cursor.lastrowid
        
        conn.commit()
        return result

def execute_transaction(operations, timeout=30.0):
    """
    Execute multiple operations in a single transaction
    operations: list of (query, params) tuples
    """
    with db_manager.get_connection(timeout=timeout) as conn:
        cursor = conn.cursor()
        results = []
        
        try:
            for query, params in operations:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                results.append(cursor.lastrowid)
            
            conn.commit()
            return results
            
        except Exception as e:
            conn.rollback()
            raise e

# Test function
def test_database_manager():
    """Test the database manager"""
    try:
        # Test simple query
        result = execute_query("SELECT COUNT(*) FROM disease_cases", fetch='one')
        print(f"✓ Database manager test successful: {result[0]} cases found")
        return True
    except Exception as e:
        print(f"✗ Database manager test failed: {e}")
        return False

if __name__ == "__main__":
    test_database_manager()