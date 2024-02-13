from codes.base_class.setting import setting

class Settings(setting):
    fold = None
    
    def run(self, step):
        # load dataset
        loaded_data = self.dataset.load()

        # run learning methods
        self.method.data = loaded_data
        learned_result = self.method.run(step)
        return None