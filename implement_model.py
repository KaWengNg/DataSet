import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the training datasets
df_train = pd.read_csv('https://raw.githubusercontent.com/KaWengNg/DataSet/main/train_OutOfDomain.csv', usecols=['sentence', 'label'], encoding='ISO-8859-1')
df_train_ec = pd.read_csv('https://raw.githubusercontent.com/KaWengNg/DataSet/main/train_EC.csv', usecols=['sentence', 'label'], encoding='ISO-8859-1')
df_train_lms = pd.read_csv('https://raw.githubusercontent.com/KaWengNg/DataSet/main/train_LMS.csv', usecols=['sentence', 'label'], encoding='ISO-8859-1')

# Concatenate the training datasets
df_train = pd.concat([df_train, df_train_ec, df_train_lms], ignore_index=True)

# Load the validation datasets
df_validate = pd.read_csv('https://raw.githubusercontent.com/KaWengNg/DataSet/main/validate_OutOfDomain.csv', usecols=['sentence', 'label'], encoding='ISO-8859-1')
df_validate_ec = pd.read_csv('https://raw.githubusercontent.com/KaWengNg/DataSet/main/validate_EC.csv', usecols=['sentence', 'label'], encoding='ISO-8859-1')
df_validate_lms = pd.read_csv('https://raw.githubusercontent.com/KaWengNg/DataSet/main/validate_LMS.csv', usecols=['sentence', 'label'], encoding='ISO-8859-1')

# Concatenate the validation datasets
df_validate = pd.concat([df_validate, df_validate_ec, df_validate_lms], ignore_index=True)

# Define the custom label mapping
custom_define_label = {
    0: "This is out of domain query.",
    1001: "What is my basic information?",
    1002: "How to edit employee information?",
    1003: "How to export employee data?",
    1004: "Can you provide the total race count for each company?",
    1005: "How do I navigate to the import employee information page?",
    1006: "Can you provide details about the employee's current job position?",
    1007: "What should I do if I forgot my current INet password?",
    1008: "How do I register a new employee in the system?",
    1009: "What is preventing me from deleting the employee?",
    2000: "What is my total leave entitlement?",
    2001: "What is my emergency leave entitlement?",
    2002: "What document do I need to attach in order to apply medical leave?",
    2026: "Why is my applied leave still pending and not approved?",
    2027: "How to cancel a leave request once it has been approved?",
    2032: "What is my emergency leave balance?",
    2037: "How many public holidays are there this month?",
    2038: "What are the public holidays observed by the company for the current year?",
    2039: "What are the upcoming scheduled public holidays for this year?",
    2040: "What are the details of my applied leave?",
    2041: "What are my pending leave applications?",
    2042: "How many leave applications are approved?",
    2043: "What are my canceling leave applications?",
    2044: "What are my rejected leave applications?",
    2045: "What are my withdrawn leave applications?",
    2046: "What are my canceled leave applications?",
    2047: "Do you have information about my colleagues?",
    2049: "How many annual leave balances will be brought forward to next year?",
    2050: "What are the details of my requested leaves?",
    2051: "What is leave module?",
    2052: "How to create a new public holiday?",
    2054: "How to add leave balance to a user?",
    2055: "How to set an expiration date on replacement leave?",
    2056: "What are the details of my upcoming leave?",
    2057: "What are the upcoming public holidays and leave schedules?",
    2058: "When is the best time to take leave?",
    2059: "What is my available leave balance that can be brought forward after forfeiture?",
    2060: "What is my leave application status?",
    2061: "What could be the reasons for my leave balance showing a negative value?",
    2062: "What could be the reasons for my brought forward leave showing a negative value?",
    2063: "What is my current actual leave balance for different leave types?",
    2064: "How to calculate my leave balance available?",
    2065: "How to calculate my leave entitlement?",
    2066: "What is the leave application status I applied on certain date?",
    2067: "How to apply for leave?",
    2068: "What types of leaves are available in the leave module?",
    2069: "What is my current leave balance for different leave types?",
    2071: "What should I do if I need to make changes to a leave application that has already been submitted or approved?",
    2083: "Does the company allow employees to carry forward unused leave days to the next year?",
    2084: "What types of documentation are required to support different leave applications according to company policies?",
    2085: "How to view my leave application history?",
    2091: "How to apply for leave balance encashment?",
    2093: "How can an administrator apply replacement leave for employees in the system?",
    2094: "How to change the setting for hourly leave?",
    2095: "How to apply for leave accumulation?",
    2096: "How can I view the leave taken by others?",
    2106: "How to perform a payroll posting for leave?",
    2107: "What are the key settings and privileges that an Human Resource Admin can manage within the leave management system?",
    2108: "Who are the approvers for my leave application?",
    2118: "How many days in advance can I apply for advance leave?",
    2119: "How can I submit a medical certificate when applying for medical leave?",
    2123: "Can I apply for sick leave for the past absences?",
    2149: "Is the specified date a company holiday, and am I entitled to take that day off?",
    2150: "How is the deduction for each day of unpaid leave calculated?",
    2151: "How can I withdraw a leave request that has not yet been approved?",
    2163: "When is the next public holiday?",
    2164: "How many leave days have I taken this year, and where can I find the details?",
    2165: "How many leave days can be brought forward to the next year?",
    2167: "What are the public holiday dates for this year?",
    2169: "What is the difference between withdraw leave and cancel leave?",
    2170: "What are the leave types I am entitled to?",
    2172: "How can I create a new leave code?",
    2173: "How many days of hospitalisation leave do I have remaining?",
    2174: "What is emergency leave?",
    2176: "How can I apply for replacement leave?",
    2177: "How can I change the leave entitlement settings?",
    2178: "How to edit leave entitlement?",
    2180: "What is the difference between Leave Encashment and Balance Encashment?",
    2181: "What could be the reasons why my leave entitlement balance is not as I expected?",
    2183: "What steps should I take if I find that my leave balance is incorrect?",
    2184: "How can I calculate unpaid leave for my long leave using my current annual leave balance?",
    2185: "How many leave requests are pending your approval today?",
    2186: "Why might the approval and applicant not receive leave application email notifications?",
    2188: "How to generate day code?",
    2189: "How do different leave types earn their leave days?",
    2191: "Why don't I have any leave entitlement showing in my record?",
    2193: "Why can't I apply for Annual Leave?",
    2194: "Why can't I apply for leave even though I've earned leave",
    2195: "What are the minimum leave entitlements in Malaysia as per the Employment Act 1955?",
    2197: "What are the rules and restrictions on the types of leaves that can be requested through the system?",
    2203: "What happens to my pending leave application if my supervisor leaves the company?",
    2204: "Does transferring to a new department impact my leave entitlement or balance?",
    2205: "Why can't I apply for marriage leave?",
    2206: "What happens to my unused Annual Leave when I resign?",
    2207: "How will I know if my leave request has been approved?",
    2209: "How to add a public holiday in system?",
    2210: "How to calculate my leave earned on specific date?",
    2211: "What’s the difference between lump sum and day-by-day methods for payroll posting?",
    2212: "What will be my annual leave entitlement after few years employment?",
    2214: "How can I view the leave taken by my subordinates?",
    2215: "What additional leave entitlements or benefits are available for long-serving employees?",
    2216: "How can I verify that all public holidays are properly set up in the system?",
    2217: "Why does my approved leave still show as pending in the system?",
    2219: "Can we buy back annual leave?",
    2221: "How to edit the leave code?"
}

# Create a continuous label mapping
unique_labels = pd.concat([df_train['label'], df_validate['label']]).unique()

label2id = {int(label): i for i, label in enumerate(sorted(unique_labels))}
id2label = {i: int(label) for label, i in label2id.items()}

# Map id2label back to custom_define_label
custom_id2label = {i: custom_define_label[label] for i, label in id2label.items()}

# Load the fine-tuned model
checkpoint_path = 'C:\\Users\\kaweng.ng\\AppData\\Local\\Programs\\Python\\Python38\\sentence-transformers\\all-MiniLM-L12-v2-lora-finetune\\checkpoint-8793'
model2 = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, num_labels=len(id2label), id2label=custom_id2label, label2id=label2id)
model2.eval()  # Set the model to evaluation mode

# Load the tokenizer
base_model_id = 'sentence-transformers/all-MiniLM-L12-v2'
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# Interactive input loop for generating predictions
while True:
    input_text = input("Enter your query (or type 'exit' to quit): ")
    
    # Exit condition
    if input_text.lower() == 'exit':
        break
    
    # Tokenize the input text
    inputs = tokenizer.encode(input_text, return_tensors="pt").to("cpu")
    
    # Compute logits (output predictions from the model)
    with torch.no_grad():  # Disable gradient calculations for inference
        logits = model2(inputs).logits
    
    # Get the predicted label by finding the index of the maximum logit value
    predictions = torch.argmax(logits, dim=-1)
    
    # Get the label ID and corresponding value from the dataset
    predicted_label_id = predictions.item()
    predicted_label_value = id2label[predicted_label_id]
    predicted_label_description = custom_define_label[predicted_label_value]
    
    # print(f"Predicted Label ID: {predicted_label_id}")
    print(f"Predicted Label Value: {predicted_label_value}")
    print(f"Map to Question: {predicted_label_description}\n")
