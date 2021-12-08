import calendar
import math
import pandas as pd
import time
import twstock
import requests
from datetime import datetime, timedelta
from dateutil import relativedelta
from db.Connection import session
from enum import Enum
from model.StockHistory import StockHistory
from sys import float_info
from talib import abstract

WinRate = {
    'Strategy1': {'highRsi':{'day':{},'week':{},'month':{}},'lowRsi':{'day':{},'week':{},'month':{}}},   
    'Strategy2': {'highRsi':{'day':{},'week':{},'month':{}},'lowRsi':{'day':{},'week':{},'month':{}}},
    'Strategy3': {'highRsi':{'day':{},'week':{},'month':{}},'lowRsi':{'day':{},'week':{},'month':{}}}} 

class HistoryType(Enum):
    DAY = ("0", "日", "短線")
    WEEK = ("1", "週", "中短線")
    MONTH = ("2", "月", "中長線")

class HistoryTypeTo(Enum):
    DB = 0
    HUMAN = 1
    EXPLAIN = 2  


class TwHistory:
    """TwHistory class"""
    dateFormatForTwStock = None
    dateFormat = None
    rsiDict = None
    williamsDict = None
    macdDict = None
    bbandDict = None
    
    def __init__(self):
        self.dateFormatForTwStock = "%Y/%m/%d"
        self.dateFormat = "%Y-%m-%d"

    def transformStrToDateTimeForTwStock(self, targetStr):
        return datetime.strptime(targetStr, self.dateFormatForTwStock)

    def transformStrToDateTime(self, targetStr):
        return datetime.strptime(targetStr, self.dateFormat)
        
    def transformDateTimeToStr(self, date):
        return date.strftime(self.dateFormat)
        
    def retIfNaN(self, num):
        if math.isnan(num):
            return None
        else:
            return num
            
    def createDataFrame(self, history):
        df = pd.DataFrame([h.as_simple_dict() for h in history])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df
        
    def deleteHistory(self, code, type, startDate, endDate):
        session.query(StockHistory).\
                filter(StockHistory.code == code).\
                filter(StockHistory.type == type).\
                filter(StockHistory.date >= self.transformDateTimeToStr(startDate)).\
                filter(StockHistory.date <= self.transformDateTimeToStr(endDate)).\
                delete()
        session.commit()

    def calculateRSI(self, df):
        rsi = abstract.RSI(df, timeperiod=5)
        self.rsiDict = {}
        for index, number in rsi.iteritems():
            self.rsiDict[self.transformDateTimeToStr(index)] = number

    def calculateWilliams(self, df):
            williams = abstract.WILLR(df, timeperiod=5)
            self.williamsDict = {}
            for index, number in williams.iteritems():
                self.williamsDict[self.transformDateTimeToStr(index)] = number

    def calculateMACD(self, df):
        macd = abstract.MACD(df)
        self.macdDict = {}
        for index, row in macd.iterrows():
            self.macdDict[self.transformDateTimeToStr(index)] = row

    def calculateBBAND(self, df):
        bband = abstract.BBANDS(df, timeperiod=22)
        self.bbandDict = {}
        for index, row in bband.iterrows():
            self.bbandDict[self.transformDateTimeToStr(index)] = row

    def updateHistoryTechnicalIndicator(self, history):
        date = history.date
        updateFlag = False
        if history.rsi is None:
            history.rsi = self.retIfNaN(self.rsiDict[date])
            updateFlag = updateFlag or history.rsi is not None
        if history.williams is None:
            history.williams = self.retIfNaN(self.williamsDict[date])
            updateFlag = updateFlag or history.williams is not None
        if history.macd is None:
            history.macd = self.retIfNaN(self.macdDict[date].macd)
            updateFlag = updateFlag or history.macd is not None
        if history.macdsignal is None:
            history.macdsignal = self.retIfNaN(self.macdDict[date].macdsignal)
            updateFlag = updateFlag or history.macdsignal is not None
        if history.macdhist is None:
            history.macdhist = self.retIfNaN(self.macdDict[date].macdhist)
            updateFlag = updateFlag or history.macdhist is not None
        if history.upperband is None:
            history.upperband = self.retIfNaN(self.bbandDict[date].upperband)
            updateFlag = updateFlag or history.upperband is not None
        if history.middleband is None:
            history.middleband = self.retIfNaN(self.bbandDict[date].middleband)
            updateFlag = updateFlag or history.middleband is not None
        if history.lowerband is None:
            history.lowerband = self.retIfNaN(self.bbandDict[date].lowerband)
            updateFlag = updateFlag or history.lowerband is not None
        if updateFlag:
            session.merge(history)

    def dayHistory(self):
        for k, v in twstock.codes.items():
            if self.isStockOrETF(v.type) and k == '3035':
                print("dayHistory code: " + k)
                dayType = self.translate(HistoryType.DAY, HistoryTypeTo.DB)  #get type value for db
                history = session.query(StockHistory).\
                        filter(StockHistory.code == k).\
                        filter(StockHistory.type == dayType).\
                        order_by(StockHistory.date.desc()).\
                        first()
                nowDate = datetime.now()
                endDateStr = self.transformDateTimeToStr(nowDate)
                startDateStr = self.transformDateTimeToStr(self.transformStrToDateTimeForTwStock(v.start)) if history is None else history.date #如果DB撈的到相對應條件的資料，就只抓最後一天
                self.finmindtrade(k, startDateStr, endDateStr, dayType)

    def weekHistory(self):
        today = self.transformStrToDateTime(self.transformDateTimeToStr(datetime.now()))
        weekStart = today - timedelta(days=today.weekday())

        for k, v in twstock.codes.items():
            if self.isStockOrETF(v.type) and self.isHistoryExist(k):
                print("weekHistory code: " + k)
                latestHistoryWeek = session.query(StockHistory).\
                        filter(StockHistory.code == k).\
                        filter(StockHistory.type == self.translate(HistoryType.WEEK, HistoryTypeTo.DB)).\
                        order_by(StockHistory.date.desc()).\
                        first()

                startdate = self.transformStrToDateTimeForTwStock(v.start) if latestHistoryWeek is None else self.transformStrToDateTime(latestHistoryWeek.date)
                weekStartPast = startdate - timedelta(days=startdate.weekday())
                weekEndPast = weekStartPast + timedelta(days=6)

                while weekStartPast <= weekStart:
                    self.deleteHistory(k, self.translate(HistoryType.WEEK, HistoryTypeTo.DB), weekStartPast, weekEndPast)
                    historyWeek = StockHistory(code=k, type=self.translate(HistoryType.WEEK, HistoryTypeTo.DB),
                            capacity=0, turnover=0, high=0, low=float_info.max, close=0)
                    firstFlag = True
                    for historyDay in session.query(StockHistory).\
                            filter(StockHistory.code == k).\
                            filter(StockHistory.type == self.translate(HistoryType.DAY, HistoryTypeTo.DB)).\
                            filter(StockHistory.date >= self.transformDateTimeToStr(weekStartPast)).\
                            filter(StockHistory.date <= self.transformDateTimeToStr(weekEndPast)).\
                            order_by(StockHistory.date.asc()).\
                            all():
                        historyWeek.date = self.transformDateTimeToStr(weekStartPast)
                        historyWeek.close = historyDay.close
                        historyWeek.capacity += historyDay.capacity
                        historyWeek.turnover += historyDay.turnover
                        if firstFlag:
                            historyWeek.open = historyDay.open
                            firstFlag = False
                        historyWeek.high = max(historyWeek.high, historyDay.high)
                        historyWeek.low = min(historyWeek.low, historyDay.low)
                    if not firstFlag:
                        session.merge(historyWeek)
                    weekStartPast += timedelta(days=7)
                    weekEndPast += timedelta(days=7)

                session.commit()

    def monthHistory(self):
        today = self.transformStrToDateTime(self.transformDateTimeToStr(datetime.now()))
        monthStart = today.replace(day=1)
        for k, v in twstock.codes.items():
            if self.isStockOrETF(v.type) and self.isHistoryExist(k):
                print("monthHistory code: " + k)
                latestHistoryMonth = session.query(StockHistory).\
                        filter(StockHistory.code == k).\
                        filter(StockHistory.type == self.translate(HistoryType.MONTH, HistoryTypeTo.DB)).\
                        order_by(StockHistory.date.desc()).\
                        first()

                startdate = self.transformStrToDateTimeForTwStock(v.start) if latestHistoryMonth is None else self.transformStrToDateTime(latestHistoryMonth.date)
                monthStartPast = startdate.replace(day=1)
                monthEndPast = monthStartPast.replace(day=calendar.monthrange(monthStartPast.year, monthStartPast.month)[1])

                while monthStartPast <= monthStart:
                    self.deleteHistory(k, self.translate(HistoryType.MONTH, HistoryTypeTo.DB), monthStartPast, monthEndPast)
                    historyMonth = StockHistory(code=k, type=self.translate(HistoryType.MONTH, HistoryTypeTo.DB),
                            capacity=0, turnover=0, high=0, low=float_info.max, close=0)
                    firstFlag = True
                    for historyDay in session.query(StockHistory).\
                            filter(StockHistory.code == k).\
                            filter(StockHistory.type == self.translate(HistoryType.DAY, HistoryTypeTo.DB)).\
                            filter(StockHistory.date >= self.transformDateTimeToStr(monthStartPast)).\
                            filter(StockHistory.date <= self.transformDateTimeToStr(monthEndPast)).\
                            order_by(StockHistory.date.asc()).\
                            all():
                        historyMonth.date = self.transformDateTimeToStr(monthStartPast)
                        historyMonth.close = historyDay.close
                        historyMonth.capacity += historyDay.capacity
                        historyMonth.turnover += historyDay.turnover
                        if firstFlag:
                            historyMonth.open = historyDay.open
                            firstFlag = False
                        historyMonth.high = max(historyMonth.high, historyDay.high)
                        historyMonth.low = min(historyMonth.low, historyDay.low)
                    if not firstFlag:
                        session.merge(historyMonth)
                    monthStartPast = monthStartPast + relativedelta.relativedelta(months=1)
                    monthEndPast = monthStartPast.replace(day=calendar.monthrange(monthStartPast.year, monthStartPast.month)[1])

                session.commit()

    def technicalIndicator(self):
        for k, v in twstock.codes.items():
            if self.isStockOrETF(v.type) and self.isHistoryExist(k):
                for historyType in HistoryType:
                    print("technicalIndicator code: " + k + ", type: " + self.translate(historyType, HistoryTypeTo.HUMAN))
                    historyList = session.query(StockHistory).\
                            filter(StockHistory.code == k).\
                            filter(StockHistory.type == self.translate(historyType, HistoryTypeTo.DB)).\
                            order_by(StockHistory.date.asc()).\
                            all()
                    if len(historyList) == 0:
                        continue
                    df = self.createDataFrame(historyList)

                    self.calculateRSI(df)
                    self.calculateWilliams(df)
                    self.calculateMACD(df)
                    self.calculateBBAND(df)

                    for history in historyList:
                        self.updateHistoryTechnicalIndicator(history)
                    session.commit()

    def diverge(self, highRsi, lowRsi, highWilliams, lowWilliams):
        turnoverDict = {}
        nameDict = {}
        
        for k, v in twstock.codes.items():
            print(k, StockHistory.code)
            if self.isStockOrETF(v.type) and self.isHistoryExist(k):
                history = session.query(StockHistory).\
                        filter(StockHistory.code == k).\
                        filter(StockHistory.type == self.translate(HistoryType.DAY, HistoryTypeTo.DB)).\
                        order_by(StockHistory.date.desc()).\
                        first()
                turnoverDict[k] = history.turnover
                nameDict[k] = v.name

        rankDict = {k: v for k, v in sorted(turnoverDict.items(), key=lambda item: item[1], reverse=True)}
        for rankIdx, code in enumerate(rankDict.keys()): #每一支股票
            for historyType in HistoryType: #日 周 月 各跑一次
                historyType = self.translate(historyType, HistoryTypeTo.DB)
                historyList = session.query(StockHistory).\
                        filter(StockHistory.code == code).\
                        filter(StockHistory.type == historyType).\
                        filter(StockHistory.rsi.isnot(None)).\
                        order_by(StockHistory.date.asc()).\
                        all()
                historyListLength = len(historyList)
                #highPeak = []
                #lowPeak = []       
                #for i in range(historyListLength):
                #    if i == 0 or i == historyListLength - 1:
                #        continue
                #    if history.rsi > historyList[i+1].rsi and history.rsi >= historyList[i-1].rsi:                     
                #        if history.rsi == historyList[i-1].rsi:
                #            self.isPreDataHighPeak(historyList, i, 1, highPeak)
                #        else:    
                #            highPeak.append(history)
                #    if history.rsi < historyList[i+1].rsi and history.rsi <= historyList[i-1].rsi:
                #        if history.rsi == historyList[i-1].rsi:
                #            self.isPreDataLowPeak(historyList, i, 1, lowPeak)
                #        else:
                #            lowPeak.append(history)                                                        
                #print(len(highPeak), len(lowPeak), highPeak[2].rsi)
                for i in range(historyListLength):
                    if historyListLength > 1 and i < historyListLength-1:
                        if historyList[i+1].high and historyList[i+1].upperband and historyList[i+1].low and historyList[i+1].lowerband:
                            if historyList[i+1].open > historyList[i+1].close and historyList[i+1].high > historyList[i+1].upperband and self.isHighRsi(highRsi, historyList[i+1].rsi) and historyList[i+1].rsi > historyList[i].rsi and historyList[i+1].williams < historyList[i].williams:
                                self.strategy1WinRate(historyList, i, "highRsi", historyType)
                            elif historyList[i+1].open < historyList[i+1].close and historyList[i+1].low < historyList[i+1].lowerband and self.isLowRsi(lowRsi, historyList[i+1].rsi) and historyList[i+1].rsi < historyList[i].rsi and historyList[i+1].williams > historyList[i].williams:
                                self.strategy1WinRate(historyList, i, "lowRsi", historyType) 
                                          
        print(WinRate['Strategy1'])                                                              

    def isPreDataHighPeak(self, historyList, i, count, highPeak):
        index = i-1
        if index-count >= 0: 
            if historyList[i].rsi == historyList[index-count].rsi:
                count = count + 1
                return self.isPreDataHighPeak(historyList, i, count, highPeak)
            if historyList[i].rsi > historyList[index-count].rsi:
                highPeak.append(historyList[i])
                
    def isPreDataLowPeak(self, historyList, i, count, lowPeak):
        index = i-1
        if index-count >= 0: 
            if historyList[i].rsi == historyList[index-count].rsi:
                count = count + 1
                return self.isPreDataLowPeak(historyList, i, count, lowPeak)
            if historyList[i].rsi < historyList[index-count].rsi:
                lowPeak.append(historyList[i])                
                
    def isStockOrETF(self, type):
        return type == "股票" or type == "ETF"

    def isHistoryExist(self, code):
        if code=='3035':
            return session.query(StockHistory).\
                    filter(StockHistory.code == code).\
                    filter(StockHistory.type == self.translate(HistoryType.DAY, HistoryTypeTo.DB)).\
                    filter(StockHistory.date == "2021-10-01").\
                    first() is not None
        return False                    

    def isHighRsi(self, highRsi, rsiVal):
        if rsiVal < highRsi:
            return False
        return True

    def isLowRsi(self, lowRsi, rsiVal):
        if rsiVal > lowRsi:
            return False
        return True

    def translate(self, historyType, historyTypeTo):
        return historyType.value[historyTypeTo.value]

    def finmindtrade(self, code, start, end, dayType):
        url = "https://api.finmindtrade.com/api/v4/data"
        parameter = {
            "dataset": "TaiwanStockPrice",
            "data_id": code,
            "start_date": start,
            "end_date": end,
            "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyMS0xMC0wMSAxNjoyMzoyNSIsInVzZXJfaWQiOiJtY3VpdGVhbGxlbiIsImlwIjoiMTE4LjE2My4xNDcuMTgyIn0.vXMykagq4kOKGrKOusgfAR3yhgcri0N_Wpe1Nb4DOiA"
        }
        resp = requests.get(url, params=parameter)
        json = resp.json()
        if json is not None:
            for data in resp.json()["data"]:
                history = StockHistory(code=code, type=dayType, date=data["date"],
                        capacity=data["Trading_Volume"], turnover=data["Trading_money"],
                        open=data["open"], high=data["max"], low=data["min"], close=data["close"])
                session.merge(history)
            session.commit()
        time.sleep(6.1)

    def histroyTypeToStr(self, historyType):
        if historyType == '0':
            return 'day'
        elif historyType == '1':
            return 'week'
        else:
            return 'month'

    def getIntervalArr(self, historyType):
        if historyType == '0':
            return [1,2,3,4,5,10,15,20,25,30,35,40]
        elif historyType == '1':
            return [1,2,4,8,12,24,48]
        else:
            return [1,2,3,4,5,6,7,8,9,10,11,12]
        
    def statistic(self, value):
        total = 0
        for data in value:
            total = total + data['rVal']
        if total > 0:
            return total/len(value)
        else:
            return 0
        
    def addRecord(self, afterVal, rsiType, timeType, data, rateType, date):
        WinRate['Strategy1'][rsiType][timeType]['after'+str(data)]['returnRate_' + rateType].append({'date': date, 'rVal': float(round(afterVal*100, 2))})    
        WinRate['Strategy1'][rsiType][timeType]['after'+str(data)]['avgReturnRate_' + rateType] = round(self.statistic(WinRate['Strategy1'][rsiType][timeType]['after'+str(data)]['returnRate_' + rateType]), 2)    
        
        
    def debuggerResult(self):
        high = len(WinRate['Strategy1']['highRsi']['week']['after1']['returnRate'])/WinRate['Strategy1']['highRsi']['week']['after1']['total']
        #low = len(WinRate['Strategy1']['lowRsi']['week']['after2']['returnRate'])/WinRate['Strategy1']['lowRsi']['week']['after2']['total']
        print('High: ' + str(int(high*100))+"%" )            
        
    def strategy1WinRate(self, historyList, i, rsiType, historyType): 
        timeType = self.histroyTypeToStr(historyType)
        afterArr = self.getIntervalArr(historyType)
        lenHistoryList = len(historyList)
        for data in afterArr: 
            bool = 'after'+str(data) in WinRate['Strategy1'][rsiType][timeType].keys() 
            if  bool == False:
                WinRate['Strategy1'][rsiType][timeType]['after'+str(data)] = {'total': 0, 'winRate': 0, 'returnRate_p': [], 'avgReturnRate_p': 0, 'returnRate_n': [], 'avgReturnRate_n': 0}              
            if i < lenHistoryList-(2+data): #基準需先保留2位 再加上要回測的天數 如果index有小於保留範圍則可進行勝率運算 
                afterVal = (historyList[i+2].open - historyList[i+2+data].close)/historyList[i+2].open
                tmpTotal = WinRate['Strategy1'][rsiType][timeType]['after'+str(data)]['total'] + 1
                WinRate['Strategy1'][rsiType][timeType]['after'+str(data)]['total'] = tmpTotal
                if afterVal < 0 and rsiType == 'lowRsi':
                    #if historyList[i].date == '2005-04-11' and historyType == '1':
                    #    print(afterVal, historyList[i+2].open, historyList[i+2+data].close)
                    self.addRecord(afterVal * -1 , rsiType, timeType, data, 'p', historyList[i].date)  
                elif afterVal > 0 and rsiType == 'lowRsi':     
                    self.addRecord(afterVal, rsiType, timeType, data, 'n', historyList[i].date)    
                if afterVal > 0 and rsiType == 'highRsi':
                    self.addRecord(afterVal, rsiType, timeType, data, 'p', historyList[i].date)  
                elif afterVal < 0 and rsiType == 'highRsi':
                    self.addRecord(afterVal * -1 , rsiType, timeType, data, 'n', historyList[i].date)  
                tmpValList = WinRate['Strategy1'][rsiType][timeType]['after'+str(data)]['returnRate_p']
                tmpRate = len(tmpValList)/tmpTotal
                WinRate['Strategy1'][rsiType][timeType]['after'+str(data)]['winRate'] = int(tmpRate*100)                    
    
twHistory = TwHistory()
#twHistory.diverge(90, 10, -20, -80)
#twHistory.diverge(80, 20, -20, -80)
twHistory.diverge(75, 25, -20, -80)
#twHistory.diverge(70, 30, -20, -80)