# AI-Based Analytics for Hollister & Coloplast Ostomy Care Products

This project analyzes 9 years of user-generated discussions (2016–2025) about Hollister and Coloplast ostomy care products using topic modeling, sentiment and emotion analysis, and interactive Power BI dashboards.

## Project Overview

This repository contains an end-to-end pipeline for:
- Scraping Reddit posts and UOAA forum discussions
- Cleaning and preparing textual data
- Running topic modeling to identify key product attributes
- Performing sentiment and emotion analysis using Google Vertex AI (Gemini 2.5)
- Visualizing insights using Power BI dashboards

The goal is to understand which product attributes drive user sentiment and how the two brands compare over time.

## Repository Structure
<structure omitted for brevity in this file>

## Data Collection
Includes over 18,000 posts and comments from Reddit and UOAA forums, scraped using Reddit API (PRAW) and BeautifulSoup, covering 2016–2025.

## Topic Modeling
Topic modeling on 15,000+ cleaned posts identified attributes like leakage, adhesives, comfort, durability, odor control, pricing, and customer support.

## Sentiment and Emotion Analysis
Using Vertex AI (Gemini 2.5), sentiment and emotion classification mapped each post to its closest product attribute.

## Power BI Dashboard
Includes brand comparison dashboards, sentiment distribution, attribute-level analysis, and yearly trends.

## Tech Stack
Python, BeautifulSoup, Reddit API, Jupyter, Vertex AI, Scikit-learn, Pandas, NumPy, Power BI.

## Running the Project
Instructions for cloning, installing dependencies, running notebooks, and opening dashboards.

## Author
Sujith Thota
