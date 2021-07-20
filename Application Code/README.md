# Application Code Description

### An overview of- Code components and their functionality

#### VIEWS.PY FILE METHODS

##### STORE METHOD

This method is the major method which does the major part of converting the raw data from the extension into a proper dataset. This dataset is then used to the train KNN machine learning model on the basis of the user's searched laptop's price, processor, ram and then we calculate the priorities and train the model. Then as per the training we pass the whole dell dataset to this model and get the proiority for all the laptops and then present the best 12 laptops which are closest to the user's action. There is a lot going in this method apart from the training of the ML model. We further refine the data after getting the predicted output to make it more meaningful. The data generated from the extension is refined in such a way that the data from amazon, google and flipkart tally accordingly. The best 12 laptops are then again sorted according to their conversion rate and then their recommendation count increases which later contributes to the data analytics part.

##### CART METHOD

This method is the fetching the last added elements in the cart and the adding the newly added element to the cart and the renders the webpage with all the laptops in the cart.
In this method we also show the most preferable accessories with each laptop. The accessories are shown according to the price of the laptops. When the laptops are added to the cart
their selected count increases which later contributes to the data analytics part.

##### RMCART METHOD

This method is used to remove the items from the cart and remove that from the database as well.

##### CHAT(FEEDBACK) METHOD

Using tflearn.regression feedback is being sentimentally analyzed and predicted whether the feedback was positive or negative and returns
a value depicting positivity and negativity in a sentence.

##### RECOMMENDATION METHOD

The chat() return values are tested against user rating and unambigous data are taken into account.
Lower the rating more changes are carried out in the recommendation page.

##### ALLLAPTOPS METHOD

All the DELL Laptops present in our database(table:catalog_laptop) will be displayed.

##### ALLACCESSORY METHOD

All the Accessories present in our database(table:catalog_accessory) will be displayed.

##### CHECKOUT METHOD

The Laptops present in the cart are removed and the checked count for respective products in the database(table:user_action) is increased.

##### ANALYTICS METHOD

Displays the following graphs:
Laptop Category vs (recommended_count,bought_count,add_to_cart_count),
Laptop Series vs recommended_count,
Product vs Conversion Rate.
Success:Out of all the Laptops recommended by the algorithm if atleast one product has been checked out it is successful.This rate has been Mapped.
Total Sales Revenue is displayed.

### An overview of- Data Models
### __Accessories table__
![alt text](/Application Code/accessory.png "accessory")
### __Cart table__
![alt text](/Application Code/cart.png "cart")
### __Laptops table__
![alt text](/Application Code/laptop.png "laptop")
### __Useractions table__
![alt text](/Application Code/useraction.png "useraction")
