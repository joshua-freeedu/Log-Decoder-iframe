import streamlit as st
import os

import time
import numpy as np
import pandas as pd

from loglizer.models import PCA
from loglizer import dataloader, preprocessing
import openai

openai.api_key = os.environ["FREEEDU_OPENAI_API_KEY"]
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
    uploaded_file = st.file_uploader("Upload a log file in csv format:",
                               type=["csv"],
                               accept_multiple_files=False)
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