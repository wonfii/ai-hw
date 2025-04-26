import pandas as pd

# 1. Створити DataFrame з цих даних та конвертувати колонку OrderDate у тип datetime.
orders = pd.read_csv('./assets/orders_sample.csv')
df = pd.DataFrame(orders)
df['OrderDate'] = pd.to_datetime(df['OrderDate'])

# 2. Додати новий стовпець TotalAmount = Quantity * Price.
df['TotalAmount'] = df['Quantity'] * df['Price']

# 3. Вивести:
total_income = df['TotalAmount'].sum() # a. Сумарний дохід магазину.
print(f"Sum: {total_income}")

average_total_amount = df['TotalAmount'].mean() # b. Середнє значення TotalAmount.
print(f"Average: {average_total_amount}") 
print()

orders_count_per_client = df['Customer'].value_counts() # c. Кількість замовлень по кожному клієнту.
print(f"Orders count per client:\n{orders_count_per_client}")
print()

# 4. Вивести замовлення, в яких сума покупки перевищує 500.
orders_over_500 = df[df['TotalAmount'] > 500]
print(f"Orders over 500:\n{orders_over_500.head(10)}")
print()

# 5. Відсортувати таблицю за OrderDate у зворотному порядку.
orders_sorted = df.sort_values(by='OrderDate', ascending = False)
print(f"Sorted orders:\n{orders_sorted.head(10)}")
print()

# 6.Вивести всі замовлення, зроблені у період з 5 по 10 червня включно.
start_date = '2023-06-05'
end_date = '2023-06-10'
orders_june = df.loc[(df['OrderDate'] >= start_date) & (df['OrderDate'] <= end_date)]
print(f"Orders from June 5 to June 10:\n{orders_june}")
print()

#7. Згрупувати замовлення за Category та підрахувати:
# a. Кількість товарів;
# b. Загальну суму продажів по кожній категорії.

grouped = df.groupby('Category').agg({
    'Quantity': 'sum',
    'TotalAmount': 'sum'
}).reset_index() 

print(grouped)
print()

# Вивести ТОП-3 клієнтів за загальною сумою покупок (TotalAmount).
top_3_clients = df.groupby('Customer')['TotalAmount'].sum().sort_values(ascending=False).head(3)
print(f"Top 3 clients:\n{top_3_clients}")