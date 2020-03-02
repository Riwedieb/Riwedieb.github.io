---
layout: post
title: Insights on Airbnb prices I found in the Airbnb Boston  dataset
date: "2020-02-22"
---

The Boston Airbnb data analyzed in the following is freely available from [kaggle](https://www.kaggle.com/airbnb/boston) and is frequently used by data scientists for practitioning. The dataset contains Airbnb activity of around 3000 providers from the years 2016 and 2017 in the Boston area. Included are e.g. informations about review scores and comments, prices, availability and location of each individual provider.  
In this case, the analysis was conducted as part of a "Data Science" course from Udacity. For completing the course, three freely selectable questions had to be answered. After a first look at the business model of Airbnb and the information available in the dataset, I choose to answer the following:
1. What is the best time to find a cheap Airbnb accommodation?
2. How do prices depend on location and neighbourhood in Boston?
3. Which parameters influence pricing most?

### Data preparation, exploration and answers to the questions
The analysis was carried out using [this](https://github.com/Riwedieb/Udacity_Boston_Analysis) Jupyter notebook.
Out of the 95 features contained in the dataset, just 60 where used for further investigations. The decision on which features to omit was done by rather common sense, e.g. features like *URLs* or host IDs which where dropped right at the beginning since they where unlikely to be useful in further analysis  

 Avoiding technical details, most of the data wrangling was transforming the data types. As an example, prices were expressed as a strings ($1,000), so transformed them into a number (1000). Categorical features, like e.g. the *host response* time or *property type* I transformed using ordinal or one-hot encoding.

### 1. What is the best time to find a cheap Airbnb accommodation?
At first, let's find out whether there is some annual time where prices are cheapest in Boston. So here is a plot of the mean prices of all host over one year:
![price_year](/images/price_over_time.png)
At first it looks like the time between January and April is the cheapest season.
But wait, to conclude something on seasonality, we would expect an annual modulation. If we look at the right end (Sep. 2017) of the plot and at the left end (Oct. 2016), we see a big jump in price from about 180$ to 250$. That's a clear non-periodic behavior, so no conclusions on seasonality can be drawn from this data.
On the other hand, there are some smaller "wiggles" visible in the plot. Maybe the wiggles correspond to a weekly modulation? Let's find out! The following plot shows the mean price calculated for each weekday:  
![price_day](/images/price_per_day.png)

Obviously there is some dependency on the day of the week.  
We can conclude, that on weekends the mean Airbnb prices are higher by about 3-4% on average. On an annual timescale on the other hand, no statements are possible based on the given dataset.

### 2. How do prices depend on location and neighborhood in Boston?
In most parts of our world, house prices depend on their location.   
So let's find out how the prices correspond location.
For each host in the dataset geographical coordinates and its neighborhood are given.  
The following map shows the location of each host, together with its color encoded price.
The more red, the more expensive the host is:  
![price_loc](/images/boston_map.jpg)

There is a clear dependency of the prices on the geographic location of the Airbnb host.
Having a closer look at the map it seems like the prices are highest in proximity to Charles river and around the Boston Public Garden.

For further investigation, we want to find out in which neighborhoods prices are highest.  
![price_neighbor](/images/price_per_neighbourhood.png)

**Conclusion:**  
The influence of the neighborhoods and host location on prices is quite clearly visible.  
As the price distribution on the map already suggested, prices are higher in the neighborhoods closer to Boston Downtown.
The lowest prices you can expect in Hyde Park and Mattapan.

### 3. Which parameters influence pricing most?
For finding out which parameters have highest influence on pricing, one common analysis method is to compute the correlation matrix. Bright colors indicate the dataset features which have a high correlation:
![price_corr](/images/sns_heatmap.png)

In the "price" column there are no strong correlations visible, except for additional fees like 'security_deposit' and 'cleaning_fee'.  
Additionally, the number of bedrooms and bathrooms has an influence on the price.    
There is one more column named 'property_type' in the dataset, which can be e.g. an apartment, house, bed & breakfast or even boat. Here the question arises how prices depend on property type. The following plot gives an overview:
![price_prop](/images/price_per_prop_type.png)
On average, prices are highest if you book a boat.
On the other hand, the plot shows the spread between prices is highest for apartments and houses. For these two property types, you can expect the broadest ranges of possible prices.  

Up to now we just investigated simple correlations between parameters and prices.  
For having a deeper look at parameters which influence prices, a random forest model was trained so we are able to investigate which feature are most important for the model.
Here are the top 5 results from the model:
1. room type: entire home/apt
2. nr. of accommodates
3. nr. of bedrooms
4. room type: private room
5. amount of cleaning fees

Especially whether you choose an entire home or a private room seems to have a big influence. The next important parameters are the number of accommodates and bedrooms and last in the list is the amount of cleaning fees you have to pay.
