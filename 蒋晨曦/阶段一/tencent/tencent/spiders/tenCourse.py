# -*- coding: utf-8 -*-
import scrapy
import tencent.items
from re import sub

class TencourseSpider(scrapy.Spider):
    name = 'tenCourse'
    allowed_domains = ['ke.qq.com']
    url = 'https://ke.qq.com/course/list?price_min=1&page=%d'
    headers = {
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.117 Safari/537.36',

    }
    page_no = 1


    def start_requests(self):
        #构建我们爬取的网页
        request = scrapy.Request(
            url=self.url % self.page_no,
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
        courses = response.xpath('//div[@data-report-module="middle-course"]/ul/li')
        for course in courses:
            item_ = tencent.items.TencentItem()

            #课程名称
            course_name = course.xpath('h4/a/text()').get()
            item_['course_name'] = '{}'.format(course_name.strip() if course_name else '')

            #课程机构
            course_organization = course.xpath('div[@class="item-line item-line--middle"]/a/text()').get()
            item_['course_organization'] = course_organization.strip() if course_organization else ''

            # 课程情况
            course_status = course.xpath('div[@class="item-line item-line--middle"]/span/text()').get()
            item_['course_status'] = course_status.strip() if course_status else ''

            #报名人数
            course_number = course.xpath('div[@class="item-line item-line--bottom"]/span[2]/text()').get().strip()
            item_['course_number'] = course_number.strip() if course_number  else ''

            #课程价格
            #class ="item-line item-line--bottom"
            #class ="line-cell item-price  custom-string"
            #item_['course_price'] = course.xpath('div/span[@class ="line-cell item-price  custom-string"]/text()').get()
            course_price = course.xpath('div[@class="item-line item-line--bottom"]/span[1]/text()').get().strip()
            #course_price_tmp = course.xpath('div[@class="item-line item-line--bottom"]/span[1]/text()').get().strip()
            #course_price = float(sub(r'[^\d.]', '', course_price_tmp))
            item_['course_price'] = course_price if course_price else ''


            output.append(item_)

        print('------------------')
        if self.page_no <= 34:
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
