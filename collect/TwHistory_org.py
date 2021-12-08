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
            if self.isStockOrETF(v.type):
                print("dayHistory code: " + k)
                dayType = self.translate(HistoryType.DAY, HistoryTypeTo.DB)
                history = session.query(StockHistory).\
                        filter(StockHistory.code == k).\
                        filter(StockHistory.type == dayType).\
                        order_by(StockHistory.date.desc()).\
                        first()
                nowDate = datetime.now()
                endDateStr = self.transformDateTimeToStr(nowDate)
                startDateStr = self.transformDateTimeToStr(self.transformStrToDateTimeForTwStock(v.start)) if history is None else history.date

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
            if self.isStockOrETF(v.type) and self.isHistoryExist(k):
                history = session.query(StockHistory).\
                        filter(StockHistory.code == k).\
                        filter(StockHistory.type == self.translate(HistoryType.DAY, HistoryTypeTo.DB)).\
                        order_by(StockHistory.date.desc()).\
                        first()
                turnoverDict[k] = history.turnover
                nameDict[k] = v.name

        rankDict = {k: v for k, v in sorted(turnoverDict.items(), key=lambda item: item[1], reverse=True)}

        print("按當日成交值由大至小排名，背離條件: rsi > " + str(highRsi) + " or rsi < " + str(lowRsi))
        for rankIdx, code in enumerate(rankDict.keys()):
            closePrice = None
            divergeDict = {}
            for historyType in HistoryType:
                historyTypeHuman = self.translate(historyType, HistoryTypeTo.HUMAN)
                historyTypeExplain = self.translate(historyType, HistoryTypeTo.EXPLAIN)
                historyList = session.query(StockHistory).\
                        filter(StockHistory.code == code).\
                        filter(StockHistory.type == self.translate(historyType, HistoryTypeTo.DB)).\
                        filter(StockHistory.rsi.isnot(None)).\
                        order_by(StockHistory.date.desc()).\
                        limit(self.recentHistoryLimit(historyType)).\
                        all()
                historyListLength = len(historyList)
                if historyListLength > 0:
                    closePrice = historyList[0].close
                if historyListLength > 1:
                    if self.isHighRsi(highRsi, historyList) and historyList[0].rsi > historyList[1].rsi and historyList[0].williams < historyList[1].williams:
                        divergeDict[historyTypeHuman + " 相鄰背離 " + historyTypeExplain + "看空"] = "rsi up williams down"
                    elif self.isLowRsi(lowRsi, historyList) and historyList[0].rsi < historyList[1].rsi and historyList[0].williams > historyList[1].williams:
                        divergeDict[historyTypeHuman + " 相鄰背離 " + historyTypeExplain + "看多"] = "rsi down williams up"
                if historyListLength > 2:
                    highPeak = []
                    lowPeak = []
                    for i, history in enumerate(historyList):
                        if i == 0 or i == historyListLength - 1:
                            continue
                        if len(highPeak) < 2 and historyList[i-1].rsi < history.rsi and history.rsi > historyList[i+1].rsi:
                            highPeak.append(history)
                        if len(lowPeak) < 2 and historyList[i-1].rsi > history.rsi and history.rsi < historyList[i+1].rsi:
                            lowPeak.append(history)
                        if len(highPeak) == 2 and len(lowPeak) == 2:
                            break
                    if len(highPeak) == 2 and self.isHighRsi(highRsi, highPeak):
                        if highPeak[0].rsi > highPeak[1].rsi and highPeak[0].williams < highPeak[1].williams:
                            divergeDict[historyTypeHuman + " 波峰背離 " + historyTypeExplain + "看空: " + highPeak[1].date + " and " + highPeak[0].date] = "rsi up williams down"
                        elif highPeak[0].rsi < highPeak[1].rsi and highPeak[0].williams > highPeak[1].williams and highPeak[0].williams >= highWilliams:
                            for low in lowPeak:
                                if highPeak[0].date > low.date and highPeak[1].date < low.date and low.williams <= lowWilliams:
                                    divergeDict[historyTypeHuman + " 波峰背離 反彈不過前高 " + historyTypeExplain + "看空: " + highPeak[1].date + " and " + highPeak[0].date] = "rsi down williams fast up"
                                    break
                    if len(lowPeak) == 2 and self.isLowRsi(lowRsi, lowPeak):
                        if lowPeak[0].rsi < lowPeak[1].rsi and lowPeak[0].williams > lowPeak[1].williams:
                            divergeDict[historyTypeHuman + " 波谷背離 " + historyTypeExplain + "看多: " + lowPeak[1].date + " and " + lowPeak[0].date] = "rsi down williams up"
                        elif lowPeak[0].rsi > lowPeak[1].rsi and lowPeak[0].williams < lowPeak[1].williams and lowPeak[0].williams <= lowWilliams:
                            for high in highPeak:
                                if lowPeak[0].date > high.date and lowPeak[1].date < high.date and high.williams >= highWilliams:
                                    divergeDict[historyTypeHuman + " 波谷背離 回測不過前低 " + historyTypeExplain + "看多: " + lowPeak[1].date + " and " + lowPeak[0].date] = "rsi up williams fast down"
                                    break

            if len(divergeDict) > 0:
                print("code: " + code + ", name: " + nameDict[code] + ", rank: " + str(rankIdx+1) + "/" + str(len(rankDict)) + ", close price: " + str(closePrice))
                for k, v in divergeDict.items():
                    print(k + " => " + v)
                print("")
        print("========================================================================================")

    def isStockOrETF(self, type):
        return type == "股票" or type == "ETF"

    def isHistoryExist(self, code):
        return session.query(StockHistory).\
                filter(StockHistory.code == code).\
                filter(StockHistory.type == self.translate(HistoryType.DAY, HistoryTypeTo.DB)).\
                filter(StockHistory.date == self.transformDateTimeToStr(datetime.now())).\
                first() is not None

    def isHighRsi(self, highRsi, historyList):
        for i, history in enumerate(historyList):
            if i < 2 and history.rsi < highRsi:
                return False
            elif i == 2:
                break
        return True

    def isLowRsi(self, lowRsi, historyList):
        for i, history in enumerate(historyList):
            if i < 2 and history.rsi > lowRsi:
                return False
            elif i == 2:
                break
        return True

    def recentHistoryLimit(self, historyType):
        if historyType == HistoryType.DAY:
            return 40
        elif historyType == HistoryType.WEEK:
            return 16
        else:
            return 6

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

twHistory = TwHistory()
twHistory.dayHistory()
twHistory.weekHistory()
twHistory.monthHistory()
twHistory.technicalIndicator()
twHistory.diverge(90, 10, -20, -80)
twHistory.diverge(80, 20, -20, -80)
twHistory.diverge(75, 25, -25, -75)