

class Classfication_Service:
    def __init__(self,predictor):
        self.predictor=predictor
        
    def predict(self,text):
        return self.predictor.predict(text)

    