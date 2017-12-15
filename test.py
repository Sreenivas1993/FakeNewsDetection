# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 16:18:29 2017

@author: shalin
"""

import RealTimeFeatureExtraction

t = "Muslims BUSTED: They Stole Millions In Govâ€™t Benefits"
b = '''Print They should pay all the back all the money plus interest. The entire family and everyone who came in with them need to be deported asap. Why did it take two years to bust them? 
Here we go again â€¦another group stealing from the government and taxpayers! A group of Somalis stole over four million in government benefits over just 10 months! 
Weâ€™ve reported on numerous cases like this one where the Muslim refugees/immigrants commit fraud by scamming our systemâ€¦Itâ€™s way out of control! More Related'''

a = RealTimeFeatureExtraction.extractFeatures(title=t, body=b)
