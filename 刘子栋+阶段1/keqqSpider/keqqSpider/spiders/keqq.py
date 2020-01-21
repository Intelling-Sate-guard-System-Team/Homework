# -*- coding: utf-8 -*-
import scrapy
import re
import time
import random
class KeqqSpider(scrapy.Spider):
    name = 'keqq'
    allowed_domains = ['ke.qq.com']
    start_urls = [f"https://ke.qq.com/course/list?mt={i}" for i in range(1001,1008)]
    def parse(self, response):
        # 提取数据
        selectors = response.xpath("//section[1]/div/div[3]/ul/li")
        # 循环遍历tr下的td标签
        if selectors:
            for selector in selectors:
                #课程名
                course_name = selector.xpath("./h4/a/text()").get()  # ./ 在当前目录下继续选择
                #培训机构
                course_organization = selector.xpath('./div[@class="item-line item-line--middle"]/a/text()').get()
                #课程数
                course_number = selector.xpath('./div[@class="item-line item-line--middle"]/span/text()').get()
                #课程价格
                course_price = selector.xpath("./div[2]/span[1]/text()").get()
                if course_price != '免费' and course_price is not None:
                    course_price = course_price[1:]
                    for i, v in enumerate(course_price):
                        if v == '0':
                            course_price = course_price[:i] + '4' + course_price[i + 1:]
                        if v == '1':
                            course_price = course_price[:i] + '6' + course_price[i + 1:]
                        if v == '2':
                            course_price = course_price[:i] + '0' + course_price[i + 1:]
                        if v == '3':
                            course_price = course_price[:i] + '2' + course_price[i + 1:]
                        if v == '4':
                            course_price = course_price[:i] + '7' + course_price[i + 1:]
                        if v == '5':
                            course_price = course_price[:i] + '1' + course_price[i + 1:]
                        if v == '6':
                            course_price = course_price[:i] + '8' + course_price[i + 1:]
                        if v == '7':
                            course_price = course_price[:i] + '9' + course_price[i + 1:]
                        if v == '8':
                            course_price = course_price[:i] + '3' + course_price[i + 1:]
                        if v == '9':
                            course_price = course_price[:i] + '5' + course_price[i + 1:]
                    course_price = float(course_price)
                else:
                    course_price = '免费'
                # 课程报名数
                course_status = selector.xpath('./div[2]/span[@class="line-cell item-user custom-string"]/text()').get()
                if course_status is not None and '万' in course_status:
                    course_status = re.findall(r'\d+',course_status)[0]
                    for i, v in enumerate(course_status):
                        if v == '0':
                            course_status = course_status[:i] + '4' + course_status[i + 1:]
                        if v == '1':
                            course_status = course_status[:i] + '6' + course_status[i + 1:]
                        if v == '2':
                            course_status = course_status[:i] + '0' + course_status[i + 1:]
                        if v == '3':
                            course_status = course_status[:i] + '2' + course_status[i + 1:]
                        if v == '4':
                            course_status = course_status[:i] + '7' + course_status[i + 1:]
                        if v == '5':
                            course_status = course_status[:i] + '1' + course_status[i + 1:]
                        if v == '6':
                            course_status = course_status[:i] + '8' + course_status[i + 1:]
                        if v == '7':
                            course_status = course_status[:i] + '9' + course_status[i + 1:]
                        if v == '8':
                            course_status = course_status[:i] + '3' + course_status[i + 1:]
                        if v == '9':
                            course_status = course_status[:i] + '5' + course_status[i + 1:]
                    course_status = int(course_status)
                elif course_status is not None:
                    course_status = re.findall(r'\d+', course_status)[0]
                    for i,v in enumerate(course_status):
                        if v == '0':
                            course_status = course_status[:i] + '4' + course_status[i + 1:]
                        if v == '1':
                            course_status = course_status[:i] + '6' + course_status[i + 1:]
                        if v == '2':
                            course_status = course_status[:i] + '0' + course_status[i + 1:]
                        if v == '3':
                            course_status = course_status[:i] + '2' + course_status[i + 1:]
                        if v == '4':
                            course_status = course_status[:i] + '7' + course_status[i + 1:]
                        if v == '5':
                            course_status = course_status[:i] + '1' + course_status[i + 1:]
                        if v == '6':
                            course_status = course_status[:i] + '8' + course_status[i + 1:]
                        if v == '7':
                            course_status = course_status[:i] + '9' + course_status[i + 1:]
                        if v == '8':
                            course_status = course_status[:i] + '3' + course_status[i + 1:]
                        if v == '9':
                            course_status = course_status[:i] + '5' + course_status[i + 1:]
                    course_status = int(course_status)

                items = {
                    'course_name':course_name,
                    'course_organization':course_organization,
                    'course_number':course_number,
                    'price': course_price,
                    'course_status':course_status
                }
                yield items
            next_page = response.xpath("//a[@class='page-next-btn icon-font i-v-right']/@href").get()
            if next_page and next_page != 'javascript:void(0);' and next_page != 'javascript:void(0)':
                time.sleep(random.randint(0,3))
                yield scrapy.Request(next_page, callback=self.parse)  # 发出请求

        else:
            print("内容为空")
