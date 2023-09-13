# To run streamlit, go to terminal and type: 'streamlit run app.py'
# Core Packages ###########################
import streamlit as st
import os

import time
import numpy as np
import pandas as pd

from loglizer.models import PCA
from loglizer import dataloader, preprocessing
import openai

openai.api_key = os.environ["FREEEDU_OPENAI_API_KEY"]

project_title = "ChatGPT Log Decoder"
project_desc = """
Intrusion Detection Systems (IDS) are powerful security tools that monitor network traffic for suspicious activity and issues alerts when such activities are discovered. 
While these systems are commonly used by cybersecurity professionals and IT experts, 
the technical jargon and analysis that an IDS provides can be incomprehensible to the average user. 
This project aims to bridge this gap by utilizing ChatGPT's advanced natural language processing to 
articulate the technical information received from an IDS into human-readable form, allowing common everyday users 
to grasp the extent of their network's security status without the need for a technical background.
"""

project_link = "https://github.com/logpai/loglizer"
project_icon = "46_Heatmap.png"
st.set_page_config(page_title=project_title, initial_sidebar_state='collapsed', page_icon=project_icon)

# additional info from the readme
add_info_md = """

# loglizer


**Loglizer is a machine learning-based log analysis toolkit for automated anomaly detection**. 
  

Logs are imperative in the development and maintenance process of many software systems. They record detailed
runtime information during system operation that allows developers and support engineers to monitor their systems and track abnormal behaviors and errors. Loglizer provides a toolkit that implements a number of machine-learning based log analysis techniques for automated anomaly detection. 

:telescope: If you use loglizer in your research for publication, please kindly cite the following paper.
+ Shilin He, Jieming Zhu, Pinjia He, Michael R. Lyu. [Experience Report: System Log Analysis for Anomaly Detection](https://jiemingzhu.github.io/pub/slhe_issre2016.pdf), *IEEE International Symposium on Software Reliability Engineering (ISSRE)*, 2016. [[Bibtex](https://dblp.org/rec/bibtex/conf/issre/HeZHL16)][[中文版本](https://github.com/AmateurEvents/article/issues/2)]
**(ISSRE Most Influential Paper)**

## Framework

![Framework of Anomaly Detection](/docs/img/framework.png)

The log analysis framework for anomaly detection usually comprises the following components:

1. **Log collection:** Logs are generated at runtime and aggregated into a centralized place with a data streaming pipeline, such as Flume and Kafka. 
2. **Log parsing:** The goal of log parsing is to convert unstructured log messages into a map of structured events, based on which sophisticated machine learning models can be applied. The details of log parsing can be found at [our logparser project](https://github.com/logpai/logparser).
3. **Feature extraction:** Structured logs can be sliced into short log sequences through interval window, sliding window, or session window. Then, feature extraction is performed to vectorize each log sequence, for example, using an event counting vector. 
4. **Anomaly detection:** Anomaly detection models are trained to check whether a given feature vector is an anomaly or not.


## Models

Anomaly detection models currently available:

| Model | Paper reference |
| :--- | :--- |
| **Supervised models** |
| LR | [**EuroSys'10**] [Fingerprinting the Datacenter: Automated Classification of Performance Crises](https://www.microsoft.com/en-us/research/wp-content/uploads/2009/07/hiLighter.pdf), by Peter Bodík, Moises Goldszmidt, Armando Fox, Hans Andersen. [**Microsoft**] |
| Decision Tree | [**ICAC'04**] [Failure Diagnosis Using Decision Trees](http://www.cs.berkeley.edu/~brewer/papers/icac2004_chen_diagnosis.pdf), by Mike Chen, Alice X. Zheng, Jim Lloyd, Michael I. Jordan, Eric Brewer. [**eBay**] |
| SVM | [**ICDM'07**] [Failure Prediction in IBM BlueGene/L Event Logs](https://www.researchgate.net/publication/4324148_Failure_Prediction_in_IBM_BlueGeneL_Event_Logs), by Yinglung Liang, Yanyong Zhang, Hui Xiong, Ramendra Sahoo. [**IBM**]|
| **Unsupervised models** |
| LOF | [**SIGMOD'00**] [LOF: Identifying Density-Based Local Outliers](), by Markus M. Breunig, Hans-Peter Kriegel, Raymond T. Ng, Jörg Sander. |
| One-Class SVM | [**Neural Computation'01**] [Estimating the Support of a High-Dimensional Distribution](), by John Platt, Bernhard Schölkopf, John Shawe-Taylor, Alex J. Smola, Robert C. Williamson. |
| Isolation Forest | [**ICDM'08**] [Isolation Forest](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf), by Fei Tony Liu, Kai Ming Ting, Zhi-Hua Zhou. |
| PCA | [**SOSP'09**] [Large-Scale System Problems Detection by Mining Console Logs](http://iiis.tsinghua.edu.cn/~weixu/files/sosp09.pdf), by Wei Xu, Ling Huang, Armando Fox, David Patterson, Michael I. Jordan. [**Intel**] |
| Invariants Mining | [**ATC'10**] [Mining Invariants from Console Logs for System Problem Detection](https://www.usenix.org/legacy/event/atc10/tech/full_papers/Lou.pdf), by Jian-Guang Lou, Qiang Fu, Shengqi Yang, Ye Xu, Jiang Li. [**Microsoft**]|
| Clustering | [**ICSE'16**] [Log Clustering based Problem Identification for Online Service Systems](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/ICSE-2016-2-Log-Clustering-based-Problem-Identification-for-Online-Service-Systems.pdf), by Qingwei Lin, Hongyu Zhang, Jian-Guang Lou, Yu Zhang, Xuewei Chen. [**Microsoft**]|
| DeepLog (coming)| [**CCS'17**] [DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning](https://www.cs.utah.edu/~lifeifei/papers/deeplog.pdf), by Min Du, Feifei Li, Guineng Zheng, Vivek Srikumar. |
| AutoEncoder (coming)| [**Arxiv'18**] [Anomaly Detection using Autoencoders in High Performance Computing Systems](https://arxiv.org/abs/1811.05269), by Andrea Borghesi, Andrea Bartolini, Michele Lombardi, Michela Milano, Luca Benini. |


## Log data
We have collected a set of labeled log datasets in [loghub](https://github.com/logpai/loghub) for research purposes. If you are interested in the datasets, please follow the link to submit your access request.

## Install
```bash
git clone https://github.com/logpai/loglizer.git
cd loglizer
pip install -r requirements.txt
```

## API usage

```python
# Load HDFS dataset. If you would like to try your own log, you need to rewrite the load function.
(x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(...)

# Feature extraction and transformation
feature_extractor = preprocessing.FeatureExtractor()
feature_extractor.fit_transform(...) 

# Model training
model = PCA()
model.fit(...)

# Feature transform after fitting
x_test = feature_extractor.transform(...)
# Model evaluation with labeled data
model.evaluate(...)

# Anomaly prediction
x_test = feature_extractor.transform(...)
model.predict(...) # predict anomalies on given data
```

For more details, please follow [the demo](./docs/demo.md) in the docs to get started. Please note that all ML models are not magic, you need to figure out how to tune the parameters in order to make them work on your own data. 

## Benchmarking results 

If you would like to reproduce the following results, please run [benchmarks/HDFS_bechmark.py](./benchmarks/HDFS_bechmark.py) on the full HDFS dataset (HDFS100k is for demo only).

|       |            | HDFS |     |
| :----:|:----:|:----:|:----:|
| **Model** | **Precision** | **Recall** | **F1** |
| LR| 0.955 |	0.911 |	0.933 |
| Decision Tree | 0.998 |	0.998 |	0.998 |
| SVM| 0.959 |	0.970 |	0.965 |
| LOF | 0.967 | 0.561 | 0.710 |
| One-Class SVM | 0.995 | 0.222| 0.363 |
| Isolation Forest |  0.830 | 0.776 | 0.802 |
| PCA | 0.975 | 0.635 | 0.769|
| Invariants Mining | 0.888 | 0.945 | 0.915|
| Clustering | 1.000 | 0.720 | 0.837 |

## Contributors
+ [Shilin He](https://shilinhe.github.io), The Chinese University of Hong Kong
+ [Jieming Zhu](https://jiemingzhu.github.io), The Chinese University of Hong Kong, currently at Huawei Noah's Ark Lab
+ [Pinjia He](https://pinjiahe.github.io/), The Chinese University of Hong Kong, currently at ETH Zurich


## Feedback
For any questions or feedback, please post to [the issue page](https://github.com/logpai/loglizer/issues/new). 

"""
#######################################################################################################################
default_log_path = os.path.join("data","HDFS","HDFS_100k.log_structured.csv")
def load_loglizer(train_log_path=default_log_path):
    print("Loading Loglizer PCA model:")
    start = time.time()
    (x_train, _), (_, _), _ = dataloader.load_HDFS(train_log_path, window='session',
                                                split_type='sequential', save_csv=True)
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf',
                                              normalization='zero-mean')
    model = PCA()
    model.fit(x_train)

    stop = time.time()
    print("Loading complete. Time elapsed: " + str(stop - start))

    # Extract the event labels for each unique eventid in the source structured log file
    # Load the CSV file
    df = pd.read_csv(train_log_path)

    # Extract unique 'EventId' and 'EventTemplate' pairs
    event_names = df[['EventId', 'EventTemplate']].drop_duplicates()
    event_names_dict = dict(zip(event_names['EventId'], event_names['EventTemplate']))

    return model, feature_extractor, event_names_dict

def analyze_with_chatgpt(anomalous_packets):
    context = {"role":"system",
               "content":"""
               You will receive 5 rows of transformed data logs, where the middlemost log is flagged as anomalous.
               Your job is to analyze and compare this log with its surrounding context logs, and find out why it was flagged as anomalous. 
               Your response should be 50 to 150 words only; be as concise as possible and address the user directly. 
               Only mention the key issues and cause of the anomaly. Never mention the Column Names, the middlemost log, surrounding context log.
               You can mention what the irregular values in a column represent, and base your report on that.
               Assume that the user has no knowledge of networks or cybersecurity. Your response should start with
                "Based on the logs,".
               Try to avoid telling the user 'if the issue persists'.
               Give simple step-by-step advice on what the user can do on their own.
               If the advice is beyond the scope of the everyday user, notify them that the assistance of a professional 
               is necessary for the level of intrusion you found.
               """}
    message_prompt = []
    message_prompt.append(context)
    anomalies = f"*Column Names*: {anomalous_packets.columns} --- "
    # Build a sequence of messages, with each message being a single packet up to the last packet flagged as anomalous
    for i,p in anomalous_packets.iterrows():
        anomalies += f"*Row {i+1}*: "
        anomalies += str(p.values)
        anomalies += " --- "

    message_format = {"role":"user",
                      "content":anomalies}
    message_prompt.append(message_format)

    # Generate the response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message_prompt,
        max_tokens=1008,
        temperature=0.7,
        n=1,
        stop=None,
    )

    # Extract the response text from the API response
    response_text = response['choices'][0]['message']['content']
    # Return the response text and updated chat history
    return response_text

def main():
    head_col = st.columns([1,8])
    with head_col[0]:
        st.image(project_icon)
    with head_col[1]:
        st.title(project_title)

    st.write(project_desc)
    st.write(f"Source Project: {project_link}")
    expander = st.expander("Additional Information on the Source Project (Loglizer)")
    expander.markdown(add_info_md)
    st.markdown("***")
    st.subheader("")
#########################################

    # instructions and file upload button
    st.subheader("""
    How to use: 
     1. Upload your log file (must be .csv format)
     2. Click the 'Run Loglizer' button
                 """)
    uploaded_file = st.file_uploader("Upload a log file in csv format:",
                               type=["csv"],
                               accept_multiple_files=False)

#########################################

    run_button = st.button("Run Loglizer")

    if "anomalies" not in st.session_state:
        st.session_state.anomalies = []
    if "col_names" not in st.session_state:
        st.session_state.col_names = []

    # button is clicked
    if run_button:
        if uploaded_file is None: # if no files were uploaded:
            st.error("Please upload a .csv log file")

        else:
            # Ensure temp directory exists
            if not os.path.exists('temp'):
                os.makedirs('temp')

            filename = os.path.basename(uploaded_file.name)
            file_path = os.path.join("temp", filename)
            # Write out the uploaded file to temp directory
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner('Loading Loglizer...'):
                model, feature_extractor, event_names_dict = load_loglizer()

            # Simulate the 'live' log feed
            (x_live, _), (_, _), _ = dataloader.load_HDFS(file_path, window='session', split_type='sequential')
            x_live, st.session_state.col_names = feature_extractor.transform(x_live)
            # st.write(f"x_live shape: {x_live.shape}")

            # Map event IDs to event templates
            st.session_state.col_names = [event_names_dict[id] if id in event_names_dict else id for id in
                                          st.session_state.col_names]

            for i in range(len(x_live)):
                log = x_live[i]
                log = np.expand_dims(log, axis=0)
                prediction = model.predict(log)
                print(f"prediction for x_live[{i}]: {prediction}")
                if prediction == 1:
                    # Anomaly detected
                    context = x_live[max(0, i - 2):min(i + 3, len(x_live))]  # Get the two logs before and after
                    st.session_state.anomalies.append(context)
            # st.write("predict on x_live:")
            # st.write(model.predict(x_live))

    st.write("Anomalous events")
    st.write(len(st.session_state.anomalies))
    #
    # st.write("Column names")
    # st.write(st.session_state.col_names)

    if len(st.session_state.anomalies) > 0:
        # Displaying the anomalous packets and sending to ChatGPT
        selected_index = st.selectbox('Select an anomaly', list(range(len(st.session_state.anomalies))), 0)
        selected_anomaly = st.session_state.anomalies[selected_index]
        st.write(f"Loglizer found and compiled {selected_anomaly.shape[1]} events from the logs.\n The middle entry has an anomaly:")
        # st.write(f"Shape: {selected_anomaly.shape} Type: {type(selected_anomaly)}")

        # Convert the numpy array to a pandas DataFrame
        selected_anomaly_df = pd.DataFrame(selected_anomaly)
        # Set the column names
        selected_anomaly_df.columns = st.session_state.col_names

        # st.write(selected_anomaly)
        st.write(selected_anomaly_df)

        if st.button('Send to ChatGPT'):
            result = analyze_with_chatgpt(selected_anomaly_df)
            st.write(result)


if __name__ == '__main__':
    main()

# To run streamlit, go to terminal and type: 'streamlit run app-source.py'
