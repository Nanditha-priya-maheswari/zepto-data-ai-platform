Zepto Data & AI Platform 
An End-to-End AI & Data Engineering Capstone

Building this platform was a massive step up from writing isolated scripts. The goal was to build a complete, connected ecosystem—taking raw web data, processing it through machine learning pipelines, and finally serving an AI-powered Support Assistant via a REST API.

This repository documents that journey, the architecture, and the roadblocks I hit (and solved) along the way.

The Journey & Challenges Faced
When building a 3-module pipeline from scratch, things rarely work perfectly on the first run. Here are the real-world engineering challenges I faced and how I resolved them:

The Version Control Trial by Fire: Pushing the final GenAI module resulted in a rejected git push due to a divergent history. I had to learn how to perform a git pull --rebase, navigate into the Vim terminal editor to save the commit, and manually resolve a merge conflict in model.py before finally syncing the codebase.

Library Deprecations on the Fly:

While building the regression model, my code broke because Scikit-learn (v1.4+) completely removed the squared=False argument for RMSE. I had to refactor the code to use the newly introduced root_mean_squared_error function.

During the LangGraph build, I hit a deprecation warning for langchain-community. I had to migrate my imports mid-build to the new standalone packages (langchain-chroma, langchain-huggingface, and langchain-core).

The Pathing Trap: I accidentally generated duplicate zepto_catalog.db SQLite files. Running the extraction script from the root directory created a database outside my data_pipeline/ folder. I had to learn how Python resolves relative paths based on the terminal's current working directory and explicitly route the connection string to data_pipeline/zepto_catalog.db.

Windows Localhost Quirks: After containerizing and serving the FastAPI app, I couldn't access it via 0.0.0.0 in my browser. I learned that Windows doesn't resolve this address in browsers, so I tested the endpoints successfully using http://localhost:8000/docs via FastAPI's Swagger UI.

 Technical Architecture & Rubric Checklist
Despite the hurdles, the final architecture hits every requirement:

Module 1: Data ETL & SQLite
Web Scraping: Extracted 60 books across 3 categories (Travel, Mystery, Historical Fiction) using requests and BeautifulSoup.

Relational DB: Designed a normalized SQLite database with strictly enforced Primary and Foreign Keys connecting Books and Categories.

SQL & Pandas: Wrote 5 distinct SQL queries covering JOIN, BETWEEN, and LIMIT, and verified the relational merge directly using pandas.merge().

Module 2: Analytics & Machine Learning
Visual EDA: Generated a 4-quadrant visual analysis (Histograms, Box Plots, Count Plots, Correlation Heatmap) using Seaborn and Pandas.

Classification: Benchmarked Logistic Regression, Decision Trees, and Random Forest.

Advanced Tuning: Integrated SMOTE (strictly on training data to prevent leakage) and ran GridSearchCV to find the best hyperparameters.

Regression: Built a multivariate linear regression pipeline to predict Fare, mapping errors via a diagnostic residual plot.

Persistence: Serialized the winning pipeline (Preprocessing + Model) using joblib into best_rf_pipeline.pkl.

Module 3: GenAI Support Assistant
Vector Retrieval: Ingested 8 Zepto policy documents, embedded them using all-MiniLM-L6-v2, and stored them in ChromaDB for Top-3 Cosine Similarity retrieval.

Agentic Workflow: Built a StateGraph using LangGraph that classifies user intent and routes questions to either a specific policy RAG retrieval or a general fallback response.

API & Deployment: Wrapped the workflow in a FastAPI application (returning Answer, Sources, and Confidence metrics) and packaged the entire environment into a production-ready Dockerfile.