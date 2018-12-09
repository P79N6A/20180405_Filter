# coding:utf-8
# author:jacobjzhang
import sys
import urllib.request as urllib2
import gevent
from gevent import monkey
monkey.patch_all()
import time
import re
from bs4 import BeautifulSoup

# 基本爬虫
def get_url_content(sUrl):
    # print 'Start to get html content of url:', sUrl

    request_timeout = 10
    request_headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'
    }

    html_content = ''
    # proxy = urllib2.ProxyHandler({'http': 'http://dev-proxy.oa.com:8080'})
    # opener = urllib2.build_opener(proxy)
    # urllib2.install_opener(opener)
    req = urllib2.Request(sUrl, None, request_headers)
    try:
        res = urllib2.urlopen(req, timeout=request_timeout)
        # print res
        time.sleep(0.5)
        html_content = res.read()
    except Exception as e:
        print('Failed to get html content of url:', sUrl)
        print(e)
        res = urllib2.urlopen(req, timeout=request_timeout)
        # print res
        time.sleep(0.5)
        html_content = res.read()

    return html_content

# 组装url
def get_base_html(word, page_num=0):
    url = ('https://zhidao.baidu.com/search?lm=0&rn=10&'
           'pn={page_num}&fr=search&ie=gbk&word={word}').format(word=urllib2.quote(word.encode('gbk')),
                                                                page_num=page_num)
    # print url
    # url = ('http://www.baidu.com/s?wd={word}').format(word=word)
    # url = ('https://zhidao.baidu.com')
    base_html = get_url_content(url)
    return base_html


# get text
def parse_base_html_text(base_html):
    #解析html文件内容
    bsHtml = BeautifulSoup(base_html, 'html.parser')
    final_data = []
    #分析返回页面正文部分中数据
    if bsHtml.find('a', attrs={'class' : 'ti'}):
        for a_data in bsHtml.find_all('a', attrs={'class': 'ti'}):
            query = a_data.get_text().strip()
            final_data.append(query)
    return final_data


# get_text_interface
def get_text_interface(word,  # 指定关键词
                       start_page=0,  # 指定起始页
                       end_page=1,  # 指定尾页
                       ):
    jobs = []
    jobs = [gevent.spawn(get_base_html, word, page_num) for page_num in range(start_page, end_page+1)]
    gevent.joinall(jobs)

    base_html_list = []
    for job in jobs:
        # print job.value
        base_html_list.append(job.value)

    text_list = []
    for base_html in base_html_list:
        text_list.extend(parse_base_html_text(base_html))

    text_list = list(set(text_list))

    return text_list






if __name__ == "__main__":
    word = '世界杯'
    text_list = get_text_interface(word, start_page=0, end_page=10)
    print(text_list)
    print('\t'.join(text_list))
    # print '\t'.join(text_list)
    # print text_list
