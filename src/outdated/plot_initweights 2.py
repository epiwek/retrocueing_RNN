#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 18:31:24 2020

@author: emilia
"""

plt.figure()

plt.subplot(121)
plt.hist(torch.reshape(model.Wrec.weight_ih_l0.detach(),(-1,)).numpy())
plt.title('w_ih0')
plt.ylabel('Counts')

plt.subplot(122)
plt.hist(torch.reshape(model.Wrec.weight_hh_l0.detach(),(-1,)).numpy())
plt.title('w_hh0')


