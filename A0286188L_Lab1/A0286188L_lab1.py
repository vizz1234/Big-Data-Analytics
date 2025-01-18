'''
Name: Vishwanath Dattatreya Doddamani
Student ID: A0286188L
NUSNET ID: E1237250
NUS Email: e1237250@u.nus.edu
'''

'''
Packages Used:
pyspark==3.5.0
Python==3.11.4
Scala==3.3.1
openjdk==11.0.22
'''

from pyspark.sql import SparkSession

#Start Spark Session
sc = SparkSession \
    .builder \
    .appName("A0286188L Lab1") \
    .getOrCreate()

'''
Step1
'''
#Reading Reviews File
reviews_df = sc.read.json("Grocery_and_Gourmet_Food.json") 
#Creating RDD and Mapping the values with (product_ID, Day) as key and (1, Rating) as Value. This is Map Operation.
reviews_df_rdd = reviews_df.rdd.map(lambda review: ((review['asin'], review['reviewTime']), (1, review['overall'])))
#This is Reduce Operation. x[0] + y[0] calculates the total number of reviews for a product on a particular day. 
#(x[1] * x[0] + y[1] * y[0])/(x[0] + y[0]) calculates the average rating using the formula (N1M1 + N2M2) / (N1 + N2). 
#Here N and M represent the number of observations and Mean respectively.
reviews_df_final = reviews_df_rdd.reduceByKey(lambda x, y: (x[0] + y[0], (x[1] * x[0] + y[1] * y[0])/(x[0] + y[0])))

'''
Step2
'''
#Reading Metadata file
meta_df = sc.read.json("meta_Grocery_and_Gourmet_Food.json")
#Dropping values which has null value for brand
meta_df = meta_df.na.drop(subset=["brand"])
#This is Map Operation for Metadata RDD. Key is Product ID and Value is Brand Name
meta_df_rdd = meta_df.rdd.map(lambda meta: (meta['asin'], meta['brand']))


'''
Step3
'''
#Joining the two RDDs based on key = Product ID. Here key of Review data RDD is tweaked to consist only Product ID as key.
integrated_data = reviews_df_final.map(lambda x : (x[0][0], (x[0][1], x[1]))).join(meta_df_rdd)


'''
Step4
'''
#Final Data is top 10 values based on average rating. 
#Data looks like this: ('B00M2OGS08', (('09 15, 2014', (505, 4.8811881188118775)), 'SURGE')) 
#x[1][0][1][0] Represents Daily Review Count (maps to 505 in the above example). Here, -x[1][0][1][0] represents ordering by reverse (descending order)
final_data = integrated_data.takeOrdered(10, key=lambda x: -x[1][0][1][0])

#Making a list of Product ID, Review Count, Average Rating, Brand Name for output
productID = [p[0] for p in final_data]
reviewCount = [r[1][0][1][0] for r in final_data]
averageRating = [r[1][0][1][1] for r in final_data]
brandName = [r[1][1] for r in final_data]


'''
Step5
'''
#To Write the output to a 'output.txt' file
with open('output.txt', 'w') as file:
    file.write(f"{'Product ID':<25}{'Review Count':<25}{'Average Rating':<25}{'Brand Name':<25}\n")
    for i in range(10):
        file.write(f"{str(productID[i]):<25}{str(reviewCount[i]):<25}{str(averageRating[i]):<25}{str(brandName[i]):<25}\n")


#Stop the Spark Session
sc.stop()