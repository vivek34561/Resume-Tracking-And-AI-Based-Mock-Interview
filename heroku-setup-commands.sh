#!/bin/bash
# Heroku MySQL Quick Setup Commands
# Copy and paste these commands after adding ClearDB/JawsDB

echo "==================================="
echo "Heroku MySQL Quick Setup"
echo "==================================="
echo ""

# Step 1: Add MySQL add-on (choose one)
echo "Step 1: Add MySQL Database Add-on"
echo "--------------------------------"
echo "Option A - ClearDB (Free 5MB):"
echo "  heroku addons:create cleardb:ignite"
echo ""
echo "Option B - JawsDB (Free 5MB):"
echo "  heroku addons:create jawsdb:kitefin"
echo ""

# Step 2: Get the database URL
echo "Step 2: Get Database URL"
echo "------------------------"
echo "For ClearDB:"
echo "  heroku config:get CLEARDB_DATABASE_URL"
echo ""
echo "For JawsDB:"
echo "  heroku config:get JAWSDB_URL"
echo ""

# Step 3: The URL will look like this
echo "Step 3: Parse the URL"
echo "---------------------"
echo "URL Format:"
echo "  mysql://USERNAME:PASSWORD@HOST:3306/DATABASE"
echo ""
echo "Example:"
echo "  mysql://b123abc:x456def@us-cdbr-east.cleardb.com:3306/heroku_abc123def456"
echo "  ├─ USER: b123abc"
echo "  ├─ PASS: x456def"
echo "  ├─ HOST: us-cdbr-east.cleardb.com"
echo "  └─ DB:   heroku_abc123def456"
echo ""

# Step 4: Set config variables
echo "Step 4: Set Environment Variables"
echo "----------------------------------"
echo "Copy these commands and replace with YOUR values:"
echo ""
echo "heroku config:set MYSQL_HOST=your-host.cleardb.com"
echo "heroku config:set MYSQL_PORT=3306"
echo "heroku config:set MYSQL_USER=your-username"
echo "heroku config:set MYSQL_PASSWORD=your-password"
echo "heroku config:set MYSQL_DATABASE=your-database-name"
echo "heroku config:set GROQ_API_KEY=your-groq-api-key"
echo ""

# Step 5: Verify
echo "Step 5: Verify Configuration"
echo "-----------------------------"
echo "heroku config"
echo ""

# Step 6: Deploy
echo "Step 6: Deploy"
echo "--------------"
echo "git add ."
echo "git commit -m 'Add Heroku MySQL support'"
echo "git push heroku main"
echo ""

# Step 7: Check logs
echo "Step 7: Check Application Logs"
echo "-------------------------------"
echo "heroku logs --tail"
echo ""

# Step 8: Open app
echo "Step 8: Open Your App"
echo "---------------------"
echo "heroku open"
echo ""

echo "==================================="
echo "Need Help?"
echo "==================================="
echo "See: HEROKU_MYSQL_SETUP.md"
echo ""
