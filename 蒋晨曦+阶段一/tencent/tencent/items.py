# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class TencentItem(scrapy.Item):
    # 课程名称
    course_name = scrapy.Field()
    # 培训机构
    course_organization = scrapy.Field()
    # 课程状态
    course_status = scrapy.Field()
    # 报名人数
    course_number = scrapy.Field()
    # 课程价格
    course_price = scrapy.Field()


class BossItem(scrapy.Item):
    job_name = scrapy.Field()

    company = scrapy.Field()

    job_salary = scrapy.Field()


class NonFeeCourse(scrapy.Item):
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