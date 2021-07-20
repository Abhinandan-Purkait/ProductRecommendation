# CAAR - Improved Product Recommendations

### Project Overview
----------------------------------
##### Problem

How to improve product recommendations for customers visiting dell online product store and increase the sales figures ensuring customer satisfaction.


##### Solution

* We gather customer's activity data from different sites example, flipkart, amazon and google using a browser extension (or Ad Networks if available).
* The gathered data is used to train a machine learning algorithm which then predicts the preference/priority value for all dell products in the store inventory.
* We further improve the predictions by implementing refinement algorithms based on dell's requirements to target the user with specific categories of products
* The final recommendations are shown to the user. Maximum customer satisfaction is achieved as the products are of high specifications in minimum customer budget.
* Now we track the user's interaction with the recommended products and refine the recommendations further based on the actions taken like - product added to cart or products checked out.
* Interactions with a product on dell.com are used to track product recommendation conversion ratio. This ratio is used to manipulate the recommendations and display popular products.
* The customer can provide a feedback in the form of a comment. This comment is run through a sentiment analysis algorithm and the factor is used to improve the predictions.
* We provide a dashboard for the marketing and analysis team to visualise the performance of our recommendation engine and its outcomes.

### Solution Description
----------------------------------

#### Architecture Diagram

The following diagrams show a high level view of data flow in the solution design
![alt text](/dfd.png "data flow")

![alt text](/analyticsdfd.png "anlytics data flow")
#### Technical Description

##### Technologies and libraries used
 Following are some important technologies and libraries used. See a complete list in the [requirements file](Application Code/requirements.txt).

|Technology | Version |
|---|---|
|python|3.6.5|
|Django|3.0|
|nltk|3.4.5|
|numpy|1.17.4|
|pandas|0.25.3|
|psycopg2|2.8.4|
|scikit-learn|0.22|
|scipy|1.3.3|
|sklearn|0.0|
|tensorflow|1.13.1|
|tflearn|0.3.2|
|postgreSQL|12.1 |


##### Setup/Installations required to run the solution for testing

1. Install the chrome browser extension by loading the __Chrome Extensions__ folder on chrome internet browser.
    * Activate developer mode in chrome
    * Go to extensions
    * Click on load unpacked and select the directory __Chrome Extensions__
    * Enable the extension
2. Install Python3.6.5
3. Install PostgreSQL and pgAdmin
    * create user/login role in pgAdmin with username= `delladmin` , password= `M*bmIZq(p#f7t9JJQ)N&` and make it as `superuser`
    * create a database with name dellhack and set its owner as `delladmin`
4. Open terminal inside the directory __Application Code__ and run the following commands
    * install virtualenv ` pip install virtualenv `
    * create virtualenv ` virtualenv dell_env `
    * activate the env ` /dell_env/Scripts/activate `
    * make sure that the requirements.txt file is in the present working directory
    * ` pip install -r requirements.txt `
    * ` python ` inside python console type <br/>` import nltk `<br/>` nltk.download('punkt') `
    * ` pip install -r requirements.txt ` again
5. change directory to __hack2hire__
    * ` python manage.py migrate `
6. go to pgAdmin in browser and go to database dellhack -> schemas -> catalog_laptop -> click on import/export -> select the file catalog_laptop from the __database__ folder
    * click on import then header=__yes__
    * similarly import other csv data files from database folder to pgAdmin
7. in the terminal ` python manage.py runserver `
8. __In Chrome open link localhost:8000__


### Team Members
----------------------------------

List of team member names and email IDs with their contributions

|Name|E-mail|Contributions|
|---|---|---|
|Chirag Khandelwal|chirag.kh.13@gmail.com|Chrome extension, feedback sentiment analysis|
|Abhinandan Purkait|purkaitabhinandan@gmail.com|ML model implementation, integration of all modules in django|
|Akash Sharma|akash.asharma.sharma11@gmail.com|Cleaning of gathered user data, implementation of model refinement algorithms|
|Reetam Nandi|reetamnandi@gmail.com|Build the website database, front end|

