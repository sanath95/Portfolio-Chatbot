"""Social media retrieval tools for public persona."""

from __future__ import annotations

import asyncio
import aiohttp
from datetime import datetime, timezone
from json import dumps
from os import getenv
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build


def normalize_timestamp(ts: str) -> str:
    """Normalize ISO timestamp to UTC with Z suffix."""
    ts = ts.replace("Z", "+00:00")
    dt = datetime.fromisoformat(ts)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


async def get_instagram_posts(account_info_url: str, media_url: str, account_info_fields: str, media_fields: str):
    """Fetch Instagram posts asynchronously."""
    token = getenv('INSTAGRAM_ACCESS_TOKEN', "")
    if not token:
        return dumps({"error": "Could not retrieve any content."})
    async with aiohttp.ClientSession() as session:
        acc_task = session.get(
            account_info_url,
            params={"fields": account_info_fields, "access_token": token}
        )
        media_task = session.get(
            media_url,
            params={
                "fields": media_fields,
                "access_token": token
            }
        )
        
        acc_resp, media_resp = await asyncio.gather(acc_task, media_task)
        acc_resp.raise_for_status()
        media_resp.raise_for_status()
        
        acc_data = await acc_resp.json()
        media_data = await media_resp.json()
        
        acc_info = {
            "handle": acc_data["username"],
            "media_count": acc_data["media_count"]
        }
        
        all_posts = media_data["data"]
        
        next_url = media_data.get("paging", {}).get("next")
        
        for _ in range(4):
            if not next_url:
                break
            
            async with session.get(next_url) as resp:
                resp.raise_for_status()
                page_data = await resp.json()
                all_posts.extend(page_data.get("data", []))
                next_url = page_data.get("paging", {}).get("next")
        
        posts_sorted = sorted(
            all_posts,
            key=lambda x: (x.get("like_count", 0), x.get("comments_count", 0)),
            reverse=True
        )
        
        for post in posts_sorted:
            post["timestamp"] = normalize_timestamp(post["timestamp"])
        
        return dumps({
            "account_info": acc_info,
            "media": posts_sorted
        })


async def get_youtube_videos():
    """Fetch YouTube videos with async batch processing."""
    access_token = getenv("YOUTUBE_ACCESS_TOKEN")
    refresh_token = getenv("YOUTUBE_REFRESH_TOKEN")
    client_id = getenv("YOUTUBE_CLIENT_ID")
    client_secret = getenv("YOUTUBE_CLIENT_SECRET")
    token_uri = getenv("YOUTUBE_TOKEN_URI")
    creds = Credentials(
        token=access_token,
        refresh_token=refresh_token,
        client_id=client_id,
        client_secret=client_secret,
        token_uri=token_uri
    )
    if not access_token or refresh_token or client_id or client_secret or token_uri:
        return dumps({"error": "Could not retrieve any content."})

    if not creds.valid and creds.refresh_token:
        creds.refresh(Request())

    youtube = build("youtube", "v3", credentials=creds)

    channel_resp = youtube.channels().list(
        part="snippet,statistics,contentDetails",
        mine=True
    ).execute()

    if not channel_resp.get("items"):
        return dumps({"error": "No channel found"})

    ch = channel_resp["items"][0]
    channel_info = {
        "handle": ch["snippet"]["title"],
        "description": ch["snippet"].get("description", ""),
        "subscriberCount": ch["statistics"].get("subscriberCount", "0"),
        "videoCount": ch["statistics"].get("videoCount", "0"),
    }

    uploads_playlist_id = ch["contentDetails"]["relatedPlaylists"]["uploads"]

    videos = []
    next_page = None

    while True:
        pl_resp = youtube.playlistItems().list(
            part="snippet,contentDetails",
            playlistId=uploads_playlist_id,
            maxResults=50,
            pageToken=next_page,
        ).execute()

        for item in pl_resp.get("items", []):
            vid_id = item["contentDetails"]["videoId"]
            snippet = item["snippet"]
            videos.append({
                "videoId": vid_id,
                "url": f"https://www.youtube.com/watch?v={vid_id}",
                "title": snippet.get("title"),
                "description": snippet.get("description", ""),
                "published_at": snippet.get("publishedAt", "")
            })

        next_page = pl_resp.get("nextPageToken")
        if not next_page:
            break

    video_map = {v["videoId"]: v for v in videos}
    all_ids = list(video_map.keys())
    
    for i in range(0, len(all_ids), 50):
        batch_ids = all_ids[i:i + 50]
        stats_resp = youtube.videos().list(
            part="statistics",
            id=",".join(batch_ids)
        ).execute()
        
        for item in stats_resp.get("items", []):
            vid_id = item["id"]
            stats = item.get("statistics", {})
            video_map[vid_id].update({
                "viewCount": int(stats.get("viewCount", 0)),
                "commentCount": int(stats.get("commentCount", 0)),
                "likeCount": int(stats.get("likeCount", 0))
            })

    videos_sorted = sorted(
        videos,
        key=lambda x: (x.get("viewCount", 0), x.get("likeCount", 0), x.get("commentCount", 0)),
        reverse=True
    )
    
    for video in videos_sorted:
        video["published_at"] = normalize_timestamp(video["published_at"])
    
    return dumps({
        "channel": channel_info,
        "videos": videos_sorted,
    })