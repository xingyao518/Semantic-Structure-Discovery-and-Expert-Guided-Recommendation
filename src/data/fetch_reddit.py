"""
Fetch Reddit running data via Pushshift API.

This module provides functionality to download running-related Q&A
and discussions from Reddit for use in the probabilistic models.
"""

import requests
import json
import time
from typing import List, Dict
from pathlib import Path


class RedditDataFetcher:
    """
    Fetcher for Reddit running data using Pushshift API.
    
    Note: Pushshift API may have rate limits. This is a skeleton
    implementation that can be adapted to current API endpoints.
    """
    
    def __init__(self, output_dir: str = "data/raw"):
        """
        Initialize the fetcher.
        
        Args:
            output_dir: Directory to save raw data files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://api.pushshift.io/reddit/search/submission"
        
    def fetch_posts(self, 
                   subreddit: str = "running",
                   size: int = 1000,
                   after: int = None,
                   before: int = None) -> List[Dict]:
        """
        Fetch posts from a subreddit.
        
        Args:
            subreddit: Subreddit name (e.g., "running", "AdvancedRunning")
            size: Number of posts to fetch per request
            after: Unix timestamp for earliest post
            before: Unix timestamp for latest post
            
        Returns:
            List of post dictionaries with 'title', 'selftext', 'created_utc', etc.
        """
        params = {
            "subreddit": subreddit,
            "size": size,
            "sort": "created_utc",
            "sort_type": "desc"
        }
        
        if after:
            params["after"] = after
        if before:
            params["before"] = before
            
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return []
    
    def fetch_comments(self, submission_id: str) -> List[Dict]:
        """
        Fetch comments for a submission.
        
        Args:
            submission_id: Reddit submission ID
            
        Returns:
            List of comment dictionaries
        """
        url = "https://api.pushshift.io/reddit/search/comment"
        params = {
            "link_id": submission_id,
            "size": 500
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except requests.exceptions.RequestException as e:
            print(f"Error fetching comments: {e}")
            return []
    
    def save_data(self, data: List[Dict], filename: str):
        """
        Save fetched data to JSON file.
        
        Args:
            data: List of post/comment dictionaries
            filename: Output filename
        """
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(data)} items to {filepath}")
    
    def fetch_and_save(self, 
                      subreddit: str = "running",
                      num_posts: int = 1000,
                      output_file: str = "running_posts.json"):
        """
        Fetch posts and save to file.
        
        Args:
            subreddit: Subreddit to fetch from
            num_posts: Number of posts to fetch
            output_file: Output filename
        """
        all_posts = []
        after = None
        
        while len(all_posts) < num_posts:
            posts = self.fetch_posts(subreddit=subreddit, 
                                    size=min(1000, num_posts - len(all_posts)),
                                    after=after)
            
            if not posts:
                break
                
            all_posts.extend(posts)
            
            if posts:
                after = posts[-1].get("created_utc")
            
            # Rate limiting
            time.sleep(1)
            
        self.save_data(all_posts, output_file)
        return all_posts


if __name__ == "__main__":
    fetcher = RedditDataFetcher()
    # Example usage (commented out to avoid API calls during setup):
    # fetcher.fetch_and_save(subreddit="running", num_posts=1000)

