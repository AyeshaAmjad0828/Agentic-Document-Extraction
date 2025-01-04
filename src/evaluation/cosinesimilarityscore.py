
#pip install openai
#pip install tiktoken

import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

#--------------------------------------------------------------------------------------------#

import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.LLM_utils import openai_embedding





def compute_cosine_similarity(extractedtext, filetext):
    

    if isinstance(filetext, dict):
        filetext = json.dumps(filetext)
    if isinstance(extractedtext, dict):
        extractedtext = json.dumps(extractedtext)
           
    file_embed = openai_embedding(filetext)
    output_embed = openai_embedding(extractedtext)

    embedding1 = np.array(file_embed).reshape(1, -1)
    embedding2 = np.array(output_embed).reshape(1, -1)

    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    
    return similarity



# extracted_text =  {
#   "Invoice": {
#     "Account number": "296664039561",
#     "Invoice Number": "42183017",
#     "Invoice Date": "August 3 , 2014",
#     "Bill to Address": "ATTN: iViveLabs Limited\n93B Sai Yu Chung\nYuen Long, N.T., 0000, HK",
#     "TOTAL AMOUNT DUE ON": "$4.11",
#     "Billing Period": "July 1 - July 31 , 2014"
#   },
#   "Summary": {
#     "AWS Service Charges": "$4.11",
#     "Charges": "$4.11",
#     "Credits": "$0.00",
#     "Tax": "$0.00",
#     "Total for this invoice": "$4.11"
#   },
#   "Detail": {
#     "AWS Data Transfer": "$0.01",
#     "Charges": "$0.01",
#     "VAT": "$0.00",
#     "Amazon Elastic Compute Cloud": "$1.87",
#     "Charges_2": "$1.87",
#     "VAT_2": "$0.00",
#     "Amazon Glacier": "$2.22",
#     "Charges_3": "$2.22",
#     "VAT_3": "$0.00",
#     "Amazon Simple Storage Service": "$0.01",
#     "Charges_4": "$0.01",
#     "VAT_4": "$0.00"
#   }
# }

# input="""
#                                    Amazon   Web  Services Invoice

#                                    Email or talk to us about your AWS account or bill, visit aws.amazon.com/contact-us/
#      Account number:               Invoice Summary
#      296664039561
#                                    Invoice Number:                        42183017
#                                    Invoice Date:                       August 3 , 2014
#      Bill to Address:
#      ATTN: iViveLabs Limited       TOTAL AMOUNT DUE ON August 3 , 2014     $4.11
#      93B Sai Yu Chung
#      Yuen Long, N.T., 0000, HK

#      This invoice is for the billing period July 1 - July 31 , 2014

#      Greetings from Amazon Web Services, we're writing to provide you with an electronic invoice for your use of AWS services. Additional information
#      regarding your bill, individual service charge details, and your account history are available on the Account Activity Page.
#      Summary

#        AWS Service Charges                                                  $4.11
#         Charges                                                             $4.11
#         Credits                                                             $0.00
#         Tax *                                                               $0.00

#      Total for this invoice                                                 $4.11


#      Detail
#        AWS Data Transfer                                                    $0.01
#         Charges                                                             $0.01

#         VAT **                                                              $0.00
#        Amazon Elastic Compute Cloud                                         $1.87
#         Charges                                                             $1.87
#         VAT **                                                              $0.00
#        Amazon Glacier                                                       $2.22

#         Charges                                                             $2.22
#         VAT **                                                              $0.00
#        Amazon Simple Storage Service                                        $0.01
#         Charges                                                             $0.01

#         VAT **                                                              $0.00



#      * May include estimated US sales tax, VAT, GST and CT Service Provider:
#      ** This is not a VAT invoice                          (Not to be used for payment remittance)
#      *** Check the GST statement attached at the end of this Invoice Amazon Web Services, Inc.
#      f †o Ur sd ae gt ea i als nd recurring charges for this statement period will be charged on 4 S1 e0 a tT tle er ,r y W A Av e 9 8N 1o 0rt 9h -5210, US
#      your next billing date. The amount of your actual charges for this statement
#      period may differ from the charges shown on this page. The charges
#      shown on this page do not include any additional usage charges accrued
#      during this statement period after the date you are viewing this page. Also,
#      one-time fees and subscription charges are assessed separately, on the
#      date that they occur.
#      All charges and prices are in US Dollars
#      All AWS Services are sold by Amazon Web Services, Inc.
#                                                                                1

# Summary
# AWS Service Charges $4.11
# Charges $4.11
# Credits $0.00
# Tax * $0.00
# Total for this invoice  $4.11

# Detail
# AWS Data Transfer $0.01
# Charges $0.01
# VAT ** $0.00
# Amazon Elastic Compute Cloud $1.87
# Charges $1.87
# VAT ** $0.00
# Amazon Glacier $2.22
# Charges $2.22
# VAT ** $0.00
# Amazon Simple Storage Service $0.01
# Charges $0.01
# VAT ** $0.00
# """

# a = compute_cosine_similarity(extracted_text, input)
# a









