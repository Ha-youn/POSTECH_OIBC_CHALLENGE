import requests
import json

_API_URL = "https://research-api.solarkim.com"
_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiM2dIQWtRSjlDQzRmWVJ3SnJ1VVRZIiwiaWF0IjoxNjk4NTg3MDYyLCJleHAiOjE3MDAyMzMyMDAsInR5cGUiOiJhcGlfa2V5In0.4NT3gHQSZ4lVOTZEXgvD4Mc5VVKg_YpBW42EaMcbmC0" 
_AUTH_PARAM = {"headers": {"Authorization": f"Bearer {_API_KEY}"}}

def _get(url: str):
    """
    주어진 url의 리소스를 조회한다.
    Args:
        url (str): API url
    """
    response = requests.get(url, **_AUTH_PARAM)
    return response.json()

def _post(url: str, data):
    """
    리소스 생성 데이터를 이용해서 주어진 url의 리소스를 생성한다.

    Args:
        url (str): API url
        data (dict): 리소스 생성용 데이터
    """
    response = requests.post(url, data=json.dumps(data), **_AUTH_PARAM)
    return response.json()
    
def _post_bids():
    """
    일단위 태양광 발전량 입찰. 시간별 24개의 발전량을 입찰하며 API가 호출된 시간에 따라 입찰 대상일이 결정된다. (https://research-api.solarkim.com/docs#tag/Competition-2023/operation/post_bids_cmpt_2023_bids_post 참고)
    """
    amounts = [0,0,0,0,0,0,0,1,9,24,11,11,17,25,21,10,4,1,0,0,0,0,0,0]
    success = _post(f"{_API_URL}/cmpt-2023/bids", amounts)
    print(success)

def _run():

    _post_bids()

if __name__ == "__main__": 
    _run()
