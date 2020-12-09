import json
import requests
from bs4 import BeautifulSoup
def get_temperature(ul):
    response = requests.get(ul)
    html = response.text
    # print(html)
    soup = BeautifulSoup(html, 'lxml')
    List = soup.find_all('tr')[1:]
    monthTem = []
    for tr in List:  # 每日
        tds = tr.find_all('td')
        atag = tds[0].find_all('a')
        tem = tds[2]
        max = tem.string.replace(' ', '').replace('\n', '').replace('\r', '').split('/')[0];
        min = tem.string.replace(' ', '').replace('\n', '').replace('\r', '').split('/')[1]
        # if max[0]=='-':
        max1 = int(max.split('℃')[0])
        min1 = int(min.split('℃')[0])

        data = atag[0].string.replace(' ', '').replace('\n', '').replace('\r', '').replace('年', '-').replace('月', '-').replace('日', '')
        print(data + ':' + '最低气温' + min + '  最高气温' + max)
        print()
        print('=' * 30)
        test_dict = {
            "date": data,
            "maxtem": max1,
            "mintem": min1,
            "avetem": int((min1 + max1) / 2)
        }
        monthTem.append(test_dict)

    return monthTem


if __name__ == '__main__':
    ul = 'http://www.tianqihoubao.com/lishi/beijing/month/'
    s = '201101'

    setting = []
    for year in range(9):  # 每年
        thisyear = []
        for month in range(12):  # 每月
            url = ul + s + '.html'
            thisMonth = get_temperature(url)
            thisyear.append(thisMonth)
            s = str(int(s) + 1)
        s = str(int(s) + 88)
        setting.append(thisyear)
    with open(r"data.json", 'w') as fw:
        json.dump(setting, fw)


    print("I like IoT!")

