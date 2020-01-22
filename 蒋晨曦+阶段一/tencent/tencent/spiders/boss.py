# -*- coding: utf-8 -*-
import scrapy
import tencent.items
from re import sub

class BossSpider(scrapy.Spider):
    name = 'boss'

    allowed_domains = ['https://www.zhipin.com']
    url = 'https://www.zhipin.com/c101210100/?query=C%2B%2B&page=%d&ka=page-%d'
    #url = 'https://ke.qq.com/course/list?price_min=1&page=%d'
    headers = {
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.117 Safari/537.36',

    }
    page_no = 1


    def start_requests(self):
        #构建我们爬取的网页
        request = scrapy.Request(
            url=self.url % self.page_no % self.page_no,
            callback=self.extract_course,
            method='GET',
            headers=self.headers,
            errback=self.error_handle,
        )
        return [request]

    def error_handle(self):
        print('无剩余数据')


    def extract_course(self, response):
        output = []
        print('开始处理数据')
        #xpath
        jobs = response.xpath('//*[@id="main"]/div/div[2]/div[1]/ul/li')
        for job in jobs:
            item_ = tencent.items.BossItem()

            item_['job_name'] = job.xpath('div/div/div/a/div[@class="job-title"]/span/text()').get()



            # #课程名称
            # course_name = course.xpath('h4/a/text()').get()
            # item_['course_name'] = '{}'.format(course_name.strip() if course_name else '')


            output.append(item_)

        print('------------------')
        if self.page_no <= 10:
            self.page_no += 1
            request =  scrapy.Request(
                url=self.url % self.page_no,
                callback=self.extract_course,
                method='GET',
                headers=self.headers,
                dont_filter=True,
                errback=self.error_handle,
            )
            output.append(request)

        return output
