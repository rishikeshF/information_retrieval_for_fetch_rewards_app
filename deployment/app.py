import streamlit as st
import pandas as pd
from safetensors import safe_open
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import pickle

st.title('Search offers in Fetch app')
st.markdown("""Fetch Rewards is a mobile app where you can earn free gift cards by scanning and uploading your shopping receipts. 
	You accumulate points for eligible receipts, which can be redeemed for various gift cards. It's a way to get rewards for your 
	everyday shopping.""") 
st.markdown("""
	If you type in a category (ex.diapers), this search engine will return
	a list of offers relevant to this category. You can also search using
	brand name (ex. Huggies) or a retailer name (ex.Target). This tool will
	return relevant offers related to that category, brand or retailer along
	with the similarity score representing how similar the result offer is to
	your search query.""")

bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

tensors = {}
with safe_open("embeddings.safetensors", framework="pt") as f : 
	for k in f.keys():
		tensors[k] = f.get_tensor(k)
corpus_embeddings = tensors['embedding']

with open('corpus.pickle', 'rb') as f:
	passages = pickle.load(f)


def search(query, top_k):

	query_embedding = bi_encoder.encode(query, convert_to_tensor=True)
	query_embedding = query_embedding #.cuda()
	hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
	hits = hits[0]

	cross_inp = [[query, passages[hit['corpus_id']]] for hit in hits]
	cross_scores = cross_encoder.predict(cross_inp)

	# Sort results by the cross-encoder scores
	for idx in range(len(cross_scores)):
	    hits[idx]['cross-score'] = cross_scores[idx]

	hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
	score_list, output_list = [],[]
	for hit in hits[:10]:
	  score_list.append("{:.3f}".format(hit['cross-score']))
	  temp_output = passages[hit['corpus_id']].replace("\n", " ")
	  temp_output = list(temp_output.rsplit('{'))[0].strip()
	  output_list.append(temp_output)

	dataframe = pd.DataFrame({'score': score_list, 'offers': output_list})
	dataframe.drop_duplicates(subset=['offers'], keep='first', inplace=True)
	return dataframe
	
	
with st.form("my_form"):
	query = st.text_input("Enter the brand name, category or  retailer name to search \
							for relevant offers 👇",
							placeholder = "Enter the text here")
	num = st.number_input('Manximum number of offers to display', min_value=1, max_value=10)
	
	submitted = st.form_submit_button("Submit")
	if submitted:
		df = search(query, num)
		st.dataframe(df, use_container_width=True)