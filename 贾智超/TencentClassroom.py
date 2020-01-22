import requests
import pymongo
from bs4 import BeautifulSoup


# def get_classification_url_list():

def get_course_info(category_url_list):

    course_info = {
        'Title': '',
        'Section': '',
        'Company': '',
        'Price': '',
        'UserNumber': ''
    }

    client = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
    client.admin.authenticate("root", "root", mechanism='SCRAM-SHA-1')
    mydb = client["Tencent"]
    mycol = mydb["info"]

    base_url = "https://ke.qq.com/"
    # 遍历每个类目
    for i in range(len(category_url_list)):
    # for i in range(1):
        # 获取每个类目下的页码，并遍历每页的内容
        category_url = base_url + str(category_url_list[i]['url'])
        # category_url = base_url + str(category_url_list[i]['url']) + "&page=1"
        payload = {}
        headers = {
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'zh-CN,zh;q=0.9',
            'cache-control': 'max-age=0',
            'cookie': 'pgv_pvid=3469647328; RK=DXC0SEzD80; ptcz=bc55f026045793645c7bf7c6800a02a1ddac3c1e7a2e90562566fa2c83641ee5; pgv_pvi=1930655744; pac_uid=0_5e23156d19c73; Hm_lvt_0c196c536f609d373a16d246a117fd44=1579408017; pgv_info=ssid=s6762236142; ts_refer=www.baidu.com/link; ts_uid=6765598577; _pathcode=0.9797899029164907; tdw_auin_data=-; tdw_data_testid=; tdw_data_flowid=; iswebp=1; tdw_first_visited=1; ts_last=ke.qq.com/course/list; tdw_data_sessionid=15794083898504626248701645; tdw_data_new_2={"auin":"-","sourcetype":"","sourcefrom":"","uin":"","visitor_id":"0473161020897892","url_page":"list","url_module":"","url_position":""}; tdw_data={"ver4":"www.baidu.com","ver5":"","ver6":"","refer":"www.baidu.com","from_channel":"","path":"afff-0.9797899029164907","auin":"-","uin":null,"real_uin":null}; Hm_lpvt_0c196c536f609d373a16d246a117fd44=1579408477',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-site': 'cross-site',
            'sec-fetch-user': '?1',
            'upgrade-insecure-requests': '1',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.117 Safari/537.36'
        }

        print(category_url)
        html = requests.request("GET", category_url, headers=headers, data=payload)
        soup = BeautifulSoup(html.content, "html.parser")

        # print(soup)
        page = soup.find_all("a", class_="page-btn")
        page = int(page[-1].get_text())
        for j in range(page):
            category_url = base_url + str(category_url_list[i]['url']) + "&page=" + str(j + 1)
            payload = {}
            headers = {
                'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
                'accept-encoding': 'gzip, deflate, br',
                'accept-language': 'zh-CN,zh;q=0.9',
                'cache-control': 'max-age=0',
                'cookie': 'pgv_pvid=3469647328; RK=DXC0SEzD80; ptcz=bc55f026045793645c7bf7c6800a02a1ddac3c1e7a2e90562566fa2c83641ee5; pgv_pvi=1930655744; pac_uid=0_5e23156d19c73; Hm_lvt_0c196c536f609d373a16d246a117fd44=1579408017; pgv_info=ssid=s6762236142; ts_refer=www.baidu.com/link; ts_uid=6765598577; _pathcode=0.9797899029164907; tdw_auin_data=-; tdw_data_testid=; tdw_data_flowid=; iswebp=1; tdw_first_visited=1; ts_last=ke.qq.com/course/list; tdw_data_sessionid=15794083898504626248701645; tdw_data_new_2={"auin":"-","sourcetype":"","sourcefrom":"","uin":"","visitor_id":"0473161020897892","url_page":"list","url_module":"","url_position":""}; tdw_data={"ver4":"www.baidu.com","ver5":"","ver6":"","refer":"www.baidu.com","from_channel":"","path":"afff-0.9797899029164907","auin":"-","uin":null,"real_uin":null}; Hm_lpvt_0c196c536f609d373a16d246a117fd44=1579408477',
                'sec-fetch-mode': 'navigate',
                'sec-fetch-site': 'cross-site',
                'sec-fetch-user': '?1',
                'upgrade-insecure-requests': '1',
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.117 Safari/537.36'
            }


            html = requests.request("GET", category_url, headers=headers, data=payload)
            soup = BeautifulSoup(html.content, "html.parser")
            li = soup.find_all("li", class_="course-card-item--v3 js-course-card-item")
            div = soup.find_all("div", class_="item-line item-line--bottom")
            # 获取每一个课程的信息
            for k in range(len(li)):

                course_info["Title"] = li[k].find_all("img")[0]["alt"]
                try:
                    course_info["Section"] = li[k].find_all("span", class_="line-cell item-task")[0].get_text()
                except:
                    print("null")
                course_info["Company"] = li[k].find_all("a", attrs={'class':['line-cell', 'item-source-link']})[0].get_text()

                course_info["Price"] = div[k].find_all("span", class_="line-cell item-price custom-string")[0].get_text()
                course_info["UserNumber"] = div[k].find_all("span", class_="line-cell item-user custom-string")[0].get_text()
                mycol.insert_one(course_info)
                course_info = {}


    print("Done")


if __name__ == '__main__':

    url = "https://ke.qq.com/"

    payload = {}
    headers = {
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'zh-CN,zh;q=0.9',
        'cache-control': 'max-age=0',
        'cookie': 'pgv_pvid=3469647328; RK=DXC0SEzD80; ptcz=bc55f026045793645c7bf7c6800a02a1ddac3c1e7a2e90562566fa2c83641ee5; pgv_pvi=1930655744; pac_uid=0_5e23156d19c73; Hm_lvt_0c196c536f609d373a16d246a117fd44=1579408017; pgv_info=ssid=s6762236142; ts_refer=www.baidu.com/link; ts_uid=6765598577; _pathcode=0.9797899029164907; tdw_auin_data=-; tdw_data_testid=; tdw_data_flowid=; iswebp=1; tdw_first_visited=1; ts_last=ke.qq.com/course/list; tdw_data_sessionid=15794083898504626248701645; tdw_data_new_2={"auin":"-","sourcetype":"","sourcefrom":"","uin":"","visitor_id":"0473161020897892","url_page":"list","url_module":"","url_position":""}; tdw_data={"ver4":"www.baidu.com","ver5":"","ver6":"","refer":"www.baidu.com","from_channel":"","path":"afff-0.9797899029164907","auin":"-","uin":null,"real_uin":null}; Hm_lpvt_0c196c536f609d373a16d246a117fd44=1579408477',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'cross-site',
        'sec-fetch-user': '?1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.117 Safari/537.36'
    }

    html = requests.request("GET", url, headers=headers, data=payload)
    soup = BeautifulSoup(html.content, "html.parser")
    # print(len(soup.findAll("div", class_="mod-nav__wrap-nav-side mod-nav__wrap-nav-side__adarea js-mod-category-side")))
    # print(response.text.encode('utf8'))


    div_list = soup.findAll("div", class_="mod-nav__wrap-nav-side mod-nav__wrap-nav-side__adarea js-mod-category-side")

    category_url_list = []
    for i in range(len(div_list)):
        a = div_list[i].findAll("a", class_="mod-nav__link-nav-third mod-nav__wrap-nav-third_line")
        for j in range(len(a)):
            url = a[j].attrs['href']
            title = a[j].attrs['title']
            category_url_list.append({'title': title, 'url': url})




    get_course_info(category_url_list)







