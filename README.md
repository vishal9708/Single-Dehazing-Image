# Single-Dehazing-Image
Single Dehazing Image using  Deep Learning


This project presented a encoder-decoder type neural net- work, that directly removes
haze given a single image of the scene without explicitly using the atmospheric scattering
model. We employed the usage of the GMAN and a parallel network to improve the process
of single image dehazing. We provided a quantitative and visual comparison of our model,
proving the superior quality of our model. We had evaluated our model on over 500 hazed
and dehazed image sets and compared to the models in the market, our model faired
better. Our model works for both indoor and outdoor images.




<img width="436" alt="image" src="https://user-images.githubusercontent.com/43999083/167935227-31ce030a-e9a4-4616-97b2-950d833ffafb.png">




To Run  ===>

1. pip install -r requiremts.txt
2. cd src
3. python my_example.py
4. cd ..
5. streamlit run app.py
