#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import re
import requests
import time
import torch
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
from openpyxl import Workbook
from openpyxl.drawing.image import Image
import arabic_reshaper
from bidi.algorithm import get_display
import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter import ttk
import webbrowser
import threading

# Paths
script_dir = os.path.dirname(__file__)
model_path = os.path.join(script_dir, "llama31")
font_path = os.path.join(script_dir, "NotoNaskhArabic-VariableFont_wght.ttf")

# Initialize Sentiment Analysis
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model=model_path,
    device=0 if torch.cuda.is_available() else -1
)

# Helper Functions
def open_linkedin():
    webbrowser.open("https://www.linkedin.com/in/kari-m-aher/")

def display_loading_message(message):
    loading_label.config(text=message)
    root.update_idletasks()

def remove_emojis(text):
    emoji_pattern = re.compile(
        r"[" 
        r"\U0001F600-\U0001F64F"  
        r"\U0001F300-\U0001F5FF"  
        r"\U0001F680-\U0001F6FF"  
        r"\U0001F1E0-\U0001F1FF"  
        r"\U00002500-\U00002BEF"  
        r"\U00002702-\U000027B0"
        r"\U000024C2-\U0001F251"
        r"\U0001F900-\U0001F9FF"
        r"\U0001FA70-\U0001FAFF"
        r"\U0001F000-\U0001F02F"
        r"]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r'', text).strip()

def reshape_arabic_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

def extract_video_details(video_url):
    pattern = r"https:\/\/www\.tiktok\.com\/@(?P<username>[^\/]+)\/video\/(?P<video_id>\d+)"
    match = re.match(pattern, video_url)
    if match:
        return match.group("username"), match.group("video_id")
    else:
        raise ValueError("Invalid TikTok video URL format.")

def scrape_tiktok_comments(video_id, num_comments, username):
    cursor = 0
    comments_list = []
    comments_fetched = 0

    while comments_fetched < num_comments:
        api_url = f"https://www.tiktok.com/api/comment/list/?aid=1988&aweme_id={video_id}&cursor={cursor}&count=20"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": f"https://www.tiktok.com/@{username}/video/{video_id}"
        }

        try:
            response = requests.get(api_url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                t_comments = data.get("comments", [])
                cursor = data.get("cursor", 0)

                if not t_comments:
                    break  # If no comments are returned, exit the loop

                for comment in t_comments:
                    user = comment.get("user", {}).get("nickname", "Unknown")
                    text = remove_emojis(comment.get("text", "")).strip()

                    if not user or user.lower() == "unknown" or not text:
                        continue

                    comments_list.append({"user": user, "text": text})
                    comments_fetched += 1

                    # If we have fetched the requested number of comments, break
                    if comments_fetched >= num_comments:
                        break

            else:
                messagebox.showerror("Error", "Failed to retrieve data.")
                return []
        except Exception as e:
            messagebox.showerror("Error", f"Network Error: {e}")
            return []

        time.sleep(1)

    # Check if there are fewer comments than requested and notify the user
    if comments_fetched < num_comments:
        messagebox.showinfo("Notice", f"Only {comments_fetched} comments were available.")

    return comments_list

def analyze_overall_sentiment(comments):
    scores = []
    labels = []

    for c in comments:
        text = c['text'][:512]
        result = sentiment_analyzer(text)[0]
        scores.append(result['score'])
        labels.append(result['label'])

    avg_score = sum(scores) / len(scores)
    most_common_label = max(set(labels), key=labels.count)
    return most_common_label, avg_score

def generate_wordcloud(comments):
    all_text = " ".join([reshape_arabic_text(c['text']) for c in comments])
    words = all_text.split()

    # Updated Stopwords list for English and Arabic
    custom_stopwords = set([
        # English Stopwords
        "i", "me", "my", "mine", "you", "your", "yours", "he", "him", "his", "she", "her", "hers", "it", "its",
        "we", "us", "our", "ours", "they", "them", "their", "theirs", "am", "is", "are", "was", "were", "be", "been", 
        "being", "have", "has", "had", "having", "do", "does", "did", "doing", "can", "could", "will", "would", "shall", 
        "should", "this", "that", "these", "those", "who", "what", "where", "when", "why", "how", "and", "but", "or", 
        "because", "as", "until", "while", "for", "with", "about", "against", "between", "under", "over", "through", 
        "the", "a", "an",

        # Arabic Stopwords
        "أنا", "هو", "هي", "إنت", "أنت", "أنتِ", "أنتَ", "إحنا", "هم", "هن", "هذا", "هذه", "ذلك", "تلك", "هؤلاء",
        "من", "ما", "أين", "كيف", "لماذا", "متى", "مع", "عن", "في", "إلى", "من", "على", "تحت", "فوق", "كان", "يكون", 
        "تكون", "كانت", "سيكون", "تكون", "ثم", "أيضا", "حتى", "ولكن", "أو", "لأن", "رغم", "كل", "شيء", "شيء ما",
        "لا", "نعم", "ليس", "لا أحد", "أي"
    ])

    # Removing stopwords
    filtered_words = [word for word in words if word.lower() not in custom_stopwords]

    # Frequency count after filtering stopwords
    freq = {}
    for w in filtered_words:
        freq[w] = freq.get(w, 0) + 1

    # Get top 50 frequent words
    top_50 = dict(sorted(freq.items(), key=lambda x: x[1], reverse=True)[:50])

    # Create the wordcloud
    wordcloud = WordCloud(
        font_path=font_path,
        background_color="white",
        width=800,
        height=600
    ).generate_from_frequencies(top_50)

    image_file = BytesIO()
    wordcloud.to_image().save(image_file, format='PNG')
    image_file.seek(0)

    return wordcloud, image_file

def save_to_excel(comments, label, score, wordcloud_image):
    filename = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
    if not filename:
        return

    wb = Workbook()
    ws = wb.active

    ws.append(["User", "Text"])
    for comment in comments:
        ws.append([comment['user'], comment['text']])

    ws.append(["Overall", "Average Score", label, score])

    if wordcloud_image:
        img = Image(wordcloud_image)
        img.anchor = 'E2'
        ws.add_image(img)

    wb.save(filename)
    messagebox.showinfo("Success", f"Data saved to {filename}")

def run_analysis():
    try:
        video_url = url_entry.get()
        num_comments = int(comments_entry.get())

        if "tiktok" not in video_url:
            messagebox.showerror("Error", "The URL must be a TikTok link.")
            return

        display_loading_message("Loading Data...")
        username, video_id = extract_video_details(video_url)

        comments = scrape_tiktok_comments(video_id, num_comments, username)

        if not comments:
            raise ValueError("No valid comments found.")

        display_loading_message("Analyzing Comments...")
        label, score = analyze_overall_sentiment(comments)

        display_loading_message("Generating WordCloud...")
        _, wordcloud_image = generate_wordcloud(comments)

        save_to_excel(comments, label, score, wordcloud_image)
        display_loading_message("Ready")

    except ValueError as ve:
        messagebox.showerror("Error", str(ve))
    except Exception as e:
        messagebox.showerror("Unexpected Error", str(e))

# Add right-click context menu for paste
def create_context_menu(entry_widget):
    context_menu = tk.Menu(root, tearoff=0)
    context_menu.add_command(label="Paste", command=lambda: entry_widget.event_generate("<<Paste>>"))

    def show_context_menu(event):
        context_menu.tk_popup(event.x_root, event.y_root)

    entry_widget.bind("<Button-3>", show_context_menu)

# Main program
root = tk.Tk()
root.title("TikTok Comments Analyzer")
root.geometry("600x400")

tk.Label(root, text="TikTok Video URL:", font=("Arial", 14)).pack(pady=10)
url_entry = tk.Entry(root, width=50, font=("Arial", 12))
url_entry.pack(pady=5)
create_context_menu(url_entry)

tk.Label(root, text="Number of Comments:", font=("Arial", 14)).pack(pady=10)
comments_entry = tk.Entry(root, width=10, font=("Arial", 12))
comments_entry.pack(pady=5)
create_context_menu(comments_entry)

loading_label = tk.Label(root, text="Ready", fg="green", font=("Arial", 12))
loading_label.pack(pady=5)

run_button = ttk.Button(root, text="Analyze Comments", command=run_analysis)
run_button.pack(pady=15)

root.mainloop()

