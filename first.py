import requests 

response = requests.get("https://dummyjson.com/products")

if response.status_code == 200:

    data = response.json()
    # data with rating 5
    rating_5 = [product['title'] for product in data['products'] if product['rating'] == 5] 

    for name in rating_5:
        print(name)

    #average price 
    total_price = sum([product['price'] for product in data['products']])
    average_price = total_price / len(data['products'])
    print(f"\nAverage Price: {average_price}")

else:
    print(f"Error: {response.status_code}")
    print("Failed to retrieve data.")
