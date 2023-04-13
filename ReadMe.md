# toolbox
* text embedding
* topic classification
* topic clustering


# Case study
## Back to the great inflation in 1980s 
### 1. FED meeting minutes
   FED meeting minutes in 1980s are quite standardized and structured. Analysis process:
* Search the most relevent content about a topic: embedding+similarity or chatGPT prompt
* Identify the patterns in FED expressions: chatGPT prompt
* Extract phases used to describe specific topics and manually analyze the subtle change in FED's attitude

### 2. NY times news
   Unlike the Fed meeting minutes, news articles are often more diverse in their topics and lack a clear structure or pattern in their language. Analysis procss:
* Filter the articles by date, sector, news_desk......If no given sections, use "topic classification";
* Embed all text(headline/abstract) and search queries (keywords/topics/sentences)
* Calculate the sim between queries and text
* Aggregate sim value by day
* Apply time decay function and plot the topic trend

# Apps
* Web Q&A
* File Q&A 
* Deep dive into Central_bank_speech with BERT
  Deployed in Stremalit
  pages: keyword extraction, topic trend tracker, heatmap of today's topics  
  https://jiayueyuan-kol-model-v1-beta.streamlit.app/
  
# References & useful materials
* https://github.com/openai/openai-cookbook.git
* https://python.langchain.com/en/latest/index.html#


# other apps
BloombergGPT: A Large Language Model for Finance: https://arxiv.org/abs/2303.17564




