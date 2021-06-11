
class EvaluatorBase:

    def __init__(self):
        self.best_results = None
        self.best_f1 = 0
        self.fps = []
        
        self.fith_p_tp = []
        self.fith_p_fp = []
        self.fith_p_fn = []

        self.t1results = {}
        self.t2results = ''

    def fitpredict(self, data):
        pass

    def setResults(self, t1, t2):
        self.t1results = t1
        self.t2results = t2