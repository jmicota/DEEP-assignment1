import stock_preprocessing

class Predictor:
    def __init__(self, name, model, args = None):
        """
        Constructor   
        :param name:  A name given to your predictor
        :param model: An instance of your ANN model class.
        :param parameters: An optional dictionary with parameters passed down to constructor.
        """
        self.name_ = name
        self.model_ = model
        return

    def get_name(self):
        """
        Return the name given to your predictor.   
        :return: name
        """
        return self.name_

    def get_model(self):
        """
         Return a reference to you model.
         :return: a model  
         """
        return self.model_

    def predict(self, info_company, info_quarter, info_daily, current_stock_price):
        """
        Predict, based on the most recent information, the development of the stock-prices for companies 0-2.
        :param info_company: A list of information about each company
                             (market_segment.txt  records)
        :param info_quarter: A list of tuples, with the latest quarterly information for each of the market sectors.
                             (market_analysis.txt records)
        :param info_daily: A list of tuples, with the latest daily information about each company (0-2).
                             (info.txt  records)
        :param current_stock_price: A list of floats, with the current stock prices for companies 0-2.

        :return: A Python 3-tuple with your predictions: go-up (True), not (False) [company0, company1, company2]
        """
        model = self.get_model()
        # TODO: why 'data' is has only 1 test case??
        data = stock_preprocessing.prediction_dataloaders(info_company, info_quarter, info_daily)
        # print(data)
        for X in data:
            # print("CHUJUUUUUUUUUU:", X)
            # print(X.view(-1, len(X)*len(X[0])))
            y = model(X.view(-1, len(X)*len(X[0])))
            # print("PREDICTOR RESULT: ", y)

        return True, True, True
