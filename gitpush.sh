#!/bin/bash

commit_msg=${1:-"🚀 quick commit"}
branch=$(git branch --show-current)

echo "📂 Branch: $branch"
echo "➕ Adding all files..."
git add .

echo "📝 Commiting: $commit_msg"
git commit -m "$commit_msg"

echo "🚀 Pushing to origin/$branch"
git push -u origin "$branch"

