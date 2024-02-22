#!/bin/bash

# आंकड़ा समुच्चय तैयार करना
# यह हिन्दी के लिए है। अन्य भारतीय भाषाओं के लिए आप https://github.com/AI4Bharat/IndicBERT/tree/main#indiccorp-v2 से उपयुक्त कड़ी का प्रयोग कर सकते हैं।
# Prepare the dataset
# This is for Hindi only. For other Indian languages, you can use the appropriate link from https://github.com/AI4Bharat/IndicBERT/tree/main#indiccorp-v2.
curl -r 0-499999999 https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/indic-corp-frozen-for-the-paper-oct-2022/hi.txt -o hi.txt

