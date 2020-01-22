# -*- coding: utf-8 -*-
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from tencent.items import TencentItem


class TencentItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    # 课程名称
    course_name = scrapy.Field()
    # 培训机构
    course_organization = scrapy.Field()
    # 课程连接
    course_link = scrapy.Field()
    # 报名人数
    course_number = scrapy.Field()
    # 课程状态
    course_status = scrapy.Field()
    # 课程价格
    course_price = scrapy.Field()

class TencentitemSpider(CrawlSpider):
    name = 'tencentItem'
    allowed_domains = ['ke.qq.com']
    # start_urls = ['https://ke.qq.com/course/list?mt=1001&st=2002&tt=3019&price_min=1&page=1']
    start_urls = ['https://ke.qq.com/course/list?mt=1001&page={}'.format(i) for i in range(1, 34)]

    def parse(self, response):
        result = response.xpath('//section[1]/div/div[3]/ul/li')
        items = []  # 数据项数组列表
        for course_ in result:
            # 数据项
            item_ = TencentItem()
            # 课程名称
            course_name = course_.xpath('h4/a/text()').get()
            item_['course_name'] = '{}'.format(course_name.strip() if course_name else '')

            # 培训机构 1
            course_organization = course_.xpath(
                'div[@class="item-line item-line--middle"]/a[@class="line-cell item-source-link "]/text()').get()
            item_['course_organization'] = course_organization.strip() if course_organization else ''
            # 课程连接
            course_link = course_.xpath('a/@href').get()
            item_['course_link'] = course_link.strip() if course_link else ''
            # 报名人数 1
            course_number = course_.xpath(
                'div[@class="item-line item-line--bottom"]/span[@class="line-cell item-user custom-string"]/text()').get()
            item_['course_number'] = course_number.strip() if course_number else ''
            # 课程状态
            course_status = course_.xpath('div[@class="item-line item-line--middle"]/span/text()').get()

            item_['course_status'] = course_status.strip() if course_status else ''
            # 课程价格 1
            course_price = course_.xpath('div[@class="item-line item-line--bottom"]/span/text()').get()
            item_['course_price'] = course_price.strip() if course_price else ''
            items.append(item_)
        # 返回数据项到管道
        return items