import pandas as pd
import numpy as np
from math import sqrt, pi, exp, log
import matplotlib.pyplot as plt
#############################################################

def data_import():
    column_titles = []
    for i in range(590):
        column_titles.append(f'p{i}')
    df = pd.read_csv("ML_assignment5_data.txt", sep=" ", names = column_titles)
    return df

#############################################################

def clean_data(df):
    mean_value = 0

    #Finding the mean of the column having NaN loop
    for column in df:
        mean_value = df[column].mean()

        # Replace NaNs in column with the
        # mean of values in the same column
        df[column].fillna(value=mean_value, inplace=True)
#############################################################
def normalize(df):
    ##### normailize
    # for each column, take all [data point] - [column mean]
    df=(df-df.mean())
#############################################################
def find_covarience(data):
    #compute covarience (590 x 590 M)
    return data.cov().T
#############################################################
def find_eigenvalues(cov_matrix):
    #find and test eigen_vactors
    eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
    # print("Eigenvector: \n",eigen_vectors,"\n")
    # print("Eigenvalues: \n", eigen_values, "\n")
    return eigen_values, eigen_vectors
#############################################################
def explain_varience(eigen_values):
    variance_explained = []
    for i in eigen_values:
        variance_explained.append((i/sum(eigen_values))*100)
    return variance_explained
#############################################################
def cumulative_varience(variance_explained):
    # Identifying components that explain at least 99%
    cumulative_variance_explained = np.cumsum(variance_explained)

    wanted = 0
    for i in cumulative_variance_explained:
        if i > 99:
            wanted = i
            break
    result = np.where(cumulative_variance_explained == i)
    return result[0]
   
    # print(result[0])
    # print(cumulative_variance_explained)

#############################################################
def projection(eigen_vectors, index):
    # Index using first components (because those explain more than 99%)
    projection_matrix = (eigen_vectors.T[:][:index]).T
    return projection_matrix
#############################################################
def get_projection(df, projection_matrix):
    # Getting the product of original standardized origonal data and the eigenvectors 
    df_pca = df.dot(projection_matrix)
    return df_pca
#############################################################
#Main process

# Step 1. import data
df = data_import()

# Step 2. clean data
clean_data(df)

# Step 3. normalize data
normalize(df)

# Step 4. While loop until first most variables needed are at 99%
def iterate(data):
# 4.a. covarience
    cov = find_covarience(data)

    # 4.b. eigen values 
    eigen_vals, eigen_vects = find_eigenvalues(cov)

    # 4.c. find varience and index for new projection PCA
    varience_explained = explain_varience(eigen_vals)
    index_wanted = int(cumulative_varience(varience_explained))
    projection_matrix = projection(eigen_vects, index_wanted+1)
    
    df_pca = get_projection(df, projection_matrix)
    total = 0
    # for i in range(0, 17):
    #     total += varience_explained[i]
    #     print(varience_explained[i])
    #     print(total)

    return df_pca
    
df_pca = iterate(df)
print(df_pca)
#############################################################

#############################################################

#############################################################
