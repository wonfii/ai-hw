from bs4 import BeautifulSoup

file = "./assets/eurofxref-daily.xml"

def parse_xml(xml_file):
    with open(xml_file, 'r') as file:
        content = file.read()

    soup = BeautifulSoup(content, 'xml')

    time_cube = soup.find('Cube', time=True)
    currencies = []

    for cube in time_cube.find_all('Cube'):
        if cube.get('currency') in ['GBP', 'USD']:
            currency = {
                'currency': cube.get('currency'),
                'rate': float(cube.get('rate'))
            }
            currencies.append(currency)

    return currencies

res = parse_xml(file)
for currency in res:
    print(f"Currency: {currency['currency']}, Rate: {currency['rate']}")
