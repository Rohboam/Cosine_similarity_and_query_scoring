#!/usr/bin/env python
# coding: utf-8

# In[26]:


from numpy import dot


# In[2]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[269]:


q = "New age of science"


# In[276]:


d0 = "The New age of science is just around the corner"
d1 = "Diesel is cheaper than petrol"
d2 = "Science is always making new and exciting discoveries"
d3 = "Petrol cars are cheaper than diesel cars Petrol cars are cheaper than diesel cars"
d4 = "New vendors have established shops around the corner"
d5 = "We will succeed in our journey"
d6 = "Sleeping makes your worries go away"
d7 = "Age of Machines is coming"
d8 = "Let them come, we will prevail"
d9 = "Hakuna matata"


# In[277]:


doc_list = [q, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9]


# In[278]:


def get_matrix(doc_list):
    doc_corpus=doc_list
    vec = TfidfVectorizer(stop_words='english')
    matrix=vec.fit_transform(doc_corpus)

    print("Feature Names", vec.get_feature_names())
    #print(matrix.toarray())
    arr = matrix.toarray()
    return arr    


# In[279]:


get_matrix(doc_list)


# In[280]:


def get_ranking(doc_list):
    res = []
    arr = get_matrix(doc_list)
    
    # Storing Query in seperate Vector
    q_vec = arr[0]
    
    # Calculating Cosine Similarity for each vector
    for i in range(0, len(doc_list)-1):
        score = dot(q_vec, arr[i+1])
        res.append((score , i))
    
    # Sorting with Most relevant Doc at first
    # Format = (Score, DocID)
    
    res.sort(reverse = True)
    print()
    print("Cosine Similarity Score: ", res)
    
    final_res = []
    
    # Storing docs for result
    for i in range(0, len(res)):
        final_res.append("doc"+str(res[i][1]))
    
    print()
    print("Ranked Docs: ", final_res)


# In[281]:


get_ranking(doc_list)


# In[ ]:





# In[ ]:




