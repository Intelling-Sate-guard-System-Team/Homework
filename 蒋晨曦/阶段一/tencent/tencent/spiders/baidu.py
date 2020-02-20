import scrapy

"""
scrapy爬虫引擎，调用爬虫程序，首先调用start_requests,其中返回的Request请求对象，
来决定我们需要爬取的网站页面。

如果没有重载start_requests函数，他的默认实现是从start_urls中循环获取的页面url

|- 提供start_urls
|- 重载start_requests

"""



class BaiduSpider(scrapy.Spider):
    name = 'baidu'
    #allowed_domains = ['www.baidu.com']
   # start_urls = ['https://www.baidu.com']

    def parse(self, response):
        print('数据处理， 默认的函数')
        self.log('日志输出')


    def handle_data(self, response):
        print('绑定处理函数，自定义函数')


    #引擎调用的第一个函数
    def start_requests(self):
        print('开始爬虫')
        return  [
            scrapy.Request('http://www.huanqiu.com', callback=self.handle_data),
            scrapy.Request('https://ke.qq.com'),
            scrapy.Request('https://ke.qq.com', dont_filter=True, callback=self.parse1),

        ]
       # return [scrapy.Request(self.start_urls[0])]

