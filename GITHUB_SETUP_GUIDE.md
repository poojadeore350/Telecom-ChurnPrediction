# 🚀 GitHub Setup and Deployment Guide

This guide will walk you through setting up your telecom churn prediction project on GitHub with professional best practices.

## 📋 Prerequisites

Before starting, ensure you have:
- [x] Git installed on your computer
- [x] A GitHub account
- [x] Your project files ready

## 🔧 Step 1: Install Git (if not already installed)

### Windows
1. Download Git from [git-scm.com](https://git-scm.com/download/win)
2. Run the installer with default settings
3. Open Command Prompt or PowerShell to verify: `git --version`

### macOS
```bash
# Using Homebrew
brew install git

# Or download from git-scm.com
```

### Linux
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install git

# CentOS/RHEL
sudo yum install git
```

## 🌐 Step 2: Create GitHub Repository

1. **Go to GitHub.com** and sign in
2. **Click the "+" icon** in the top right corner
3. **Select "New repository"**
4. **Fill in repository details:**
   - Repository name: `telecom-churn-prediction`
   - Description: `Advanced ML solution for predicting customer churn in telecommunications`
   - Set to **Public** (recommended for portfolio)
   - **Don't** initialize with README (we already have one)
5. **Click "Create repository"**

## 💻 Step 3: Initialize Local Git Repository

Open your terminal/command prompt in your project directory and run:

```bash
# Navigate to your project directory
cd "c:\Users\DELL\Desktop\stage d'été 2\ml_project"

# Initialize git repository
git init

# Add all files to staging
git add .

# Create initial commit
git commit -m "Initial commit: Complete telecom churn prediction project"

# Add remote origin (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/telecom-churn-prediction.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## 🔐 Step 4: Authentication Setup

### Option 1: Personal Access Token (Recommended)

1. **Go to GitHub Settings** → Developer settings → Personal access tokens → Tokens (classic)
2. **Click "Generate new token"**
3. **Set expiration** (recommend 90 days)
4. **Select scopes:**
   - [x] repo (Full control of private repositories)
   - [x] workflow (Update GitHub Action workflows)
5. **Copy the token** (you won't see it again!)
6. **Use token as password** when prompted during git push

### Option 2: SSH Key Setup

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# Add to SSH agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Copy public key to clipboard
cat ~/.ssh/id_ed25519.pub
```

Then add the public key to GitHub: Settings → SSH and GPG keys → New SSH key

## 📁 Step 5: Organize Repository Structure

Ensure your repository has this professional structure:

```
telecom-churn-prediction/
├── 📊 Data Files
│   ├── churn-bigml-80.csv
│   └── churn-bigml-20.csv
├── 📓 Notebooks
│   ├── Telecom_Customer_Churn_Prediction_Professional.ipynb
│   └── [Original notebook if needed]
├── 🐍 Source Code
│   ├── src/
│   │   ├── __init__.py
│   │   ├── data_preprocessing.py
│   │   ├── eda_utils.py
│   │   └── model_training.py
├── 🤖 Models
│   ├── models/
│   │   └── README.md
├── 📋 Documentation
│   ├── README.md
│   ├── GITHUB_SETUP_GUIDE.md
│   └── requirements.txt
└── 🔧 Configuration
    ├── .gitignore
    └── [Other config files]
```

## 🏷️ Step 6: Create Releases and Tags

After your initial push, create a release:

```bash
# Create and push a tag
git tag -a v1.0.0 -m "Initial release: Complete churn prediction solution"
git push origin v1.0.0
```

Then on GitHub:
1. Go to your repository
2. Click "Releases" → "Create a new release"
3. Select your tag (v1.0.0)
4. Add release title: "v1.0.0 - Initial Release"
5. Add description of features and improvements
6. Click "Publish release"

## 📊 Step 7: Add GitHub Actions (Optional)

Create `.github/workflows/ci.yml` for automated testing:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.8'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        python -m pytest tests/ -v
    
    - name: Check code style
      run: |
        flake8 src/ --max-line-length=88
```

## 🎨 Step 8: Enhance Repository Appearance

### Add Badges to README
Update your README.md with status badges:

```markdown
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.1+-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
```

### Create Repository Topics
On GitHub, add topics to your repository:
- machine-learning
- data-science
- churn-prediction
- telecommunications
- python
- jupyter-notebook
- scikit-learn

## 📈 Step 9: Portfolio Enhancement

### Create GitHub Pages (Optional)
1. Go to repository Settings
2. Scroll to "Pages" section
3. Select source: "Deploy from a branch"
4. Choose "main" branch and "/ (root)" folder
5. Your project will be available at: `https://yourusername.github.io/telecom-churn-prediction`

### Pin Repository
1. Go to your GitHub profile
2. Click "Customize your pins"
3. Select this repository to showcase it

## 🔄 Step 10: Ongoing Maintenance

### Regular Updates
```bash
# Make changes to your code
git add .
git commit -m "Add feature: [describe your changes]"
git push origin main
```

### Branching Strategy
```bash
# Create feature branch
git checkout -b feature/new-model
# Make changes
git add .
git commit -m "Add new model implementation"
git push origin feature/new-model
# Create pull request on GitHub
```

### Version Management
```bash
# Create new version
git tag -a v1.1.0 -m "Add advanced feature engineering"
git push origin v1.1.0
```

## 🛡️ Step 11: Security Best Practices

### Protect Sensitive Data
- Never commit API keys, passwords, or personal data
- Use environment variables for sensitive configuration
- Add sensitive files to `.gitignore`

### Repository Settings
1. Go to repository Settings
2. Enable "Restrict pushes that create files larger than 100 MB"
3. Enable "Automatically delete head branches" for clean history

## 📞 Step 12: Collaboration Setup

### Branch Protection Rules
1. Go to Settings → Branches
2. Add rule for `main` branch:
   - [x] Require pull request reviews
   - [x] Require status checks to pass
   - [x] Restrict pushes that create files larger than 100 MB

### Issue Templates
Create `.github/ISSUE_TEMPLATE/bug_report.md`:

```markdown
---
name: Bug report
about: Create a report to help us improve
title: ''
labels: bug
assignees: ''
---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Environment:**
 - OS: [e.g. Windows 10]
 - Python version: [e.g. 3.8]
 - Browser [e.g. chrome, safari]

**Additional context**
Add any other context about the problem here.
```

## ✅ Verification Checklist

Before finalizing, ensure:

- [x] Repository is public and accessible
- [x] README.md is comprehensive and professional
- [x] All code is properly documented
- [x] .gitignore excludes unnecessary files
- [x] Requirements.txt is complete and accurate
- [x] Repository has appropriate topics/tags
- [x] License is included (if applicable)
- [x] All notebooks run without errors
- [x] Code follows consistent style guidelines

## 🎉 Success!

Your professional telecom churn prediction project is now live on GitHub! 

### Next Steps:
1. **Share your project** on LinkedIn and other professional networks
2. **Add to your portfolio** website or resume
3. **Continue improving** with new features and models
4. **Engage with the community** by responding to issues and pull requests

### Portfolio Tips:
- Write a blog post about your project
- Create a video walkthrough
- Present at local meetups or conferences
- Use this project in job interviews as a showcase

---

**Need Help?** 
- GitHub Documentation: [docs.github.com](https://docs.github.com)
- Git Tutorial: [git-scm.com/docs/gittutorial](https://git-scm.com/docs/gittutorial)
- GitHub Community: [github.community](https://github.community)

**Happy coding!** 🚀