import requests
import subprocess
from bs4 import BeautifulSoup
import urllib.request, urllib.error
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from langdetect import detect
import pandas as pd

def is_english(text):
    try:
        language = detect(text)
    except Exception as e:
        print('There are no enough features!!', str(e))
        return 'unknown', False
    return language, True if language == 'en' else False

chromedriver_path = './chromedriver'

def download_text(url):
    policy_text = None
    policy_html = None
    TIMEOUT = 20
    chromeOptions = webdriver.ChromeOptions()
    chromeOptions.add_argument("--no-sandbox")
    chromeOptions.add_argument("--enable-javascript")
    chromeOptions.add_argument("--headless")
    chromeOptions.add_argument('--disable-dev-shm-usage')
    chromeOptions.add_argument("--lang=en")
    driver = webdriver.Chrome(executable_path=r'{}'.format(chromedriver_path), options=chromeOptions)
    try:
        WebDriverWait(driver, TIMEOUT).until(EC.presence_of_element_located((By.TAG_NAME, "html")))
        driver.get(url)
        element = driver.find_element_by_tag_name('html')
        policy_text = element.get_attribute('innerText')
        policy_html = driver.page_source
    except TimeoutException as e:
        reason = "HTML element has not been load after {} seconds".format(TIMEOUT)
        print(reason)
    except Exception as e:
        reason = "Error while downloading with Selenium"
        print(reason)
    finally:
        driver.close()
        return policy_text, policy_html

folder = './police_domain_top_100'

df = pd.read_excel('./domains_testing_model.xlsx')

for i in range(len(df)):
    domain = df['domain'].iloc[i]
    url = df['privacy_selenium'].iloc[i]
    try:
        policy_text, policy_html = download_text(url)
        f = open('./{}/{}.txt'.format(folder, domain), 'w')
        f.write(policy_text)
        f.close()
    except:
        pass

