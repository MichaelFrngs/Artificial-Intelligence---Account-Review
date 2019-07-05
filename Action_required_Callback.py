import os
os.chdir("C:/Users/mfrangos2016/Desktop/R/Leap Ahead Data Merger")
import pandas as pd
import datetime as dt
    
#SET FEATURE VARIABLES HERE. Which data will we use to predict bad accounts
#UNUSED: "FA_Offered", "FA_Paid",,"TotalPayments", "PercentPaid", "FA_Accepted","Balance"
predictiveVars = ["Balance","TotalFees","TotalPayments","FA_Shortage","FA_Accepted","TR","SP","FA","VA","LateOnPayment"]

#Set up the calendars
FirstPaymentDate = dt.datetime(2019,5,28)
SecondPaymentDate = dt.datetime(2019,6,22)
ThirdPaymentDate = dt.datetime(2019,7,20)

CurrentDate = dt.datetime.now()
CurrentMonth = CurrentDate.month
CurrentDay = CurrentDate.day

def calculate_payment_percentage():
    if CurrentDate > FirstPaymentDate and CurrentDate >SecondPaymentDate and CurrentDate > ThirdPaymentDate:
        PaymentThreshold = 1-.01
    elif CurrentDate > FirstPaymentDate and CurrentDate >SecondPaymentDate:
        PaymentThreshold = 2/4-.01
    elif CurrentDate > FirstPaymentDate:
        PaymentThreshold = 1/4-.01
    print("The payment threshold is: ",PaymentThreshold)
    return PaymentThreshold


def main():
    #training data
    trainingData = pd.read_csv("Action Required data.csv", encoding = "ISO-8859-1")
    
    
    ModifiedData = trainingData
    
    #Convert to strings
    ModifiedData["MethodOfPayment"] = ModifiedData["MethodOfPayment"].astype(str)
    
    #MAKE 1-HOT COLUMNS
    #Fill Tuition Reimbursement Data
    i=0
    for cell in trainingData["MethodOfPayment"]:
        if "tr" in cell.lower():
          ModifiedData.loc[i,"TR"] = 1
        else:
            ModifiedData.loc[i,"TR"] = 0
        if "fa" in cell.lower():
          ModifiedData.loc[i,"FA"] = 1
        else:
            ModifiedData.loc[i,"FA"] = 0    
        if "sp" in cell.lower():
          ModifiedData.loc[i,"SP"] = 1
        else:
            ModifiedData.loc[i,"SP"] = 0   
        if "va" in cell.lower():
          ModifiedData.loc[i,"VA"] = 1
        else:
            ModifiedData.loc[i,"VA"] = 0  
        if "va" in cell.lower():
          ModifiedData.loc[i,"VA"] = 1
        else:
            ModifiedData.loc[i,"VA"] = 0 
        i=i+1
    
    
    
    #Make the overage or underage of financial aid column
    i=0
    for cell in trainingData["MethodOfPayment"]:
        ModifiedData.loc[i,"FA_Shortage"] = ModifiedData.loc[i,"TotalFees"]- ModifiedData.loc[i,"FA_Accepted"]
        i=i+1
    

    
    #Threshold for figuring out who's late on payments
    PaymentThreshold = calculate_payment_percentage()
    
    i=0
    for cell in trainingData["MethodOfPayment"]:
        #If Student is late and selfpay
        if "sp" in cell.lower() and float(ModifiedData.loc[i,"PercentPaid"]) < PaymentThreshold and "tr" not in cell.lower() and "fa" not in cell.lower():
            ModifiedData.loc[i,"LateOnPayment"] = 1
        else:
            ModifiedData.loc[i,"LateOnPayment"] = 0
        i=i+1
    
    
    ModifiedData.to_csv("NewReport.csv")
    
    
    
    
    
    
    
    #BUILD THE NEURAL NETWORK
    predictiveData = trainingData[predictiveVars]
    
    #This section removes extranous data (not in predictive list) from the training data
    trainingData = trainingData[:][["ZNumber", "ActionRequired"]]
    trainingData = pd.concat([trainingData, predictiveData], axis = 1)
    
    
    
    ##BALANCE THE DATA
    #Section not used. Although predictions were accurate, the model was flagging more accounts than needed.
    
    #Problem = pd.DataFrame(trainingData[trainingData[:]["ActionRequired"] == 1])
    #NoProblem = pd.DataFrame(trainingData[trainingData[:]["ActionRequired"] == 0])
    #smallestSizedDf = min(len(Problem),len(NoProblem))
    #Problem = Problem[:smallestSizedDf]
    #NoProblem = NoProblem[:smallestSizedDf]
    ##Recombine the data. It is now balanced with 50% problem items and 50% good items.
    #trainingData = pd.concat([Problem, NoProblem], axis = 0)
    
    
    
    X= trainingData[:][predictiveVars]
    X['ZNumber'] = trainingData["ZNumber"]
    X.set_index("ZNumber", inplace = True)
    
    y= pd.DataFrame(trainingData[:]["ActionRequired"])
    y['ZNumber'] = trainingData["ZNumber"]
    y.set_index("ZNumber", inplace = True)
    
    
    
    
    #standardizing the input feature
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = pd.DataFrame(sc.fit_transform(X))
    X
    
    #Replace na's with 0
    X.fillna(value=0, inplace= True)
    
    
    #We now split the input features and target variables into 
    #training dataset and test dataset. out test dataset will be 30% of our entire dataset.
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    from keras import Sequential
    from keras.layers import Dense, Dropout
    
    
    classifier = Sequential()
    #First Hidden Layer
    classifier.add(Dense(8, activation='relu', kernel_initializer='random_normal', input_dim=len(predictiveVars))) #we have 11 inputs, so len shows 11
    #Second  Hidden Layer
    classifier.add(Dense(16, activation='relu', kernel_initializer='random_normal'))
    #Second  Hidden Layer
    classifier.add(Dense(64, activation='relu', kernel_initializer='random_normal'))
    classifier.add(Dense(128, activation='relu', kernel_initializer='random_normal'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(256, activation='relu', kernel_initializer='random_normal'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(512, activation='relu', kernel_initializer='random_normal'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(1024, activation='relu', kernel_initializer='random_normal'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(2048, activation='relu', kernel_initializer='random_normal'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(4096, activation='relu', kernel_initializer='random_normal'))
    classifier.add(Dropout(0.2))
    #Output Layer
    classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
    
    #Compiling the neural network
    classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
    
    
    
    
    
    #Let's make our output reporting system
    #Tensorboard is a called a "callback"
    #This "board" shows us results and stuff.
    #To access tensorboard, open cmd.exe and write: tensorboard --logdir=logs
    #If that command gives you problems, downgrade by using: pip install tensorboard==1.12.2   into anaconda promp
    #Finally, go to the posted url in the cmd prompt to open tensorboard
    from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
    NAME = f"Model {CurrentDate.month}-{CurrentDate.day}-{CurrentDate.year} {CurrentDate.hour}-{CurrentDate.minute}"
    tensorboard = TensorBoard(log_dir = f"C:/Users/mfrangos2016/Desktop/R/Leap Ahead Data Merger/logs/{NAME}") #Dynamically created names for our logs. Goes by the model name we defined above
    #More tensorboard settings
    filepath = "CallBack Model-{epoch:02d}-{val_acc:.3f}"
    checkpoint = ModelCheckpoint("models/{}.model".format(
            filepath,monitor = "val_acc",
            verbose = 1,
            save_best_only = True,
            mode = "max"))
    
    
    
    ##########################################
    #Fitting the data to the training dataset
    ##########################################
    classifier.fit(X_train,y_train, batch_size=8, epochs=1000, shuffle=True, validation_data=(X_test,y_test), callbacks = [tensorboard,checkpoint])
    
    eval_model=classifier.evaluate(X_train, y_train)
    print("Loss = ", eval_model[0],"||||||||||||||", "Accuracy = ", eval_model[1])
    
    #We now predict the output for our test dataset. If the prediction is greater than 0.5 then the output is 1 else the output is 0
    y_pred=classifier.predict(X_test)
    y_pred =(y_pred>0.5)
    
    #Now is the moment of truth. we check the accuracy on the test dataset
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    
    
    print("True Positive (no problems)", cm[0][0], "False Positive", cm[0][1],"\n",
          "False Negative", cm[1][0], "True Negative (Problems Detected)", cm[1][1])
    
    print("Negative Accuracy = ", cm[1][1]/(cm[1][1]+cm[1][0]))
    print("Positive Accuracy = ", cm[0][0]/(cm[0][0]+cm[0][1]))
    print("Total Accuracy = ", (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))
    
    
    
    
    
    
    
    
    #LETS MAKE PREDICTIONS ON AN UNSEEN DATASET
    
    #Load Data
    SecondValidationSet = pd.read_csv("FinalOutputData - All Merged Data.csv")
    
    #Data for testing bugs
    #SecondValidationSet = pd.read_csv("aCTION REquired data.csv")
    
    SecondValidationSet["MethodOfPayment"] = SecondValidationSet["MethodOfPayment"].astype(str)
    
    #Process the data into 1-hot columns
    i=0
    for cell in SecondValidationSet["MethodOfPayment"]:
        if "tr" in cell.lower():
          SecondValidationSet.loc[i,"TR"] = 1
        else:
            SecondValidationSet.loc[i,"TR"] = 0
        if "fa" in cell.lower():
          SecondValidationSet.loc[i,"FA"] = 1
        else:
            SecondValidationSet.loc[i,"FA"] = 0    
        if "sp" in cell.lower():
          SecondValidationSet.loc[i,"SP"] = 1
        else:
            SecondValidationSet.loc[i,"SP"] = 0   
        if "va" in cell.lower():
          SecondValidationSet.loc[i,"VA"] = 1
        else:
            SecondValidationSet.loc[i,"VA"] = 0  
        if "va" in cell.lower():
          SecondValidationSet.loc[i,"VA"] = 1
        else:
            SecondValidationSet.loc[i,"VA"] = 0 
            
        i=i+1
        
    i=0
    for cell in SecondValidationSet["MethodOfPayment"]:
        #If Student is late and selfpay
        if "sp" in cell.lower() and SecondValidationSet.loc[i,"PercentPaid"] < PaymentThreshold and "tr" not in cell.lower() and "fa" not in cell.lower():
            SecondValidationSet.loc[i,"LateOnPayment"] = 1
        else:
            SecondValidationSet.loc[i,"LateOnPayment"] = 0    
        i=i+1
    
    #Make the overage or underage of financial aid column
    i=0
    for cell in SecondValidationSet["MethodOfPayment"]:
        SecondValidationSet.loc[i,"FA_Shortage"] = SecondValidationSet.loc[i,"TotalFees"]- SecondValidationSet.loc[i,"FA_Accepted"]
        i=i+1
    
    
    X2= SecondValidationSet[:][predictiveVars]
    
    #standardizing the input feature
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X2 = pd.DataFrame(sc.fit_transform(X2))   
    
    #Replace na's with 0
    X2.fillna(value=0, inplace= True)
    
    X2["ZNumber"] = SecondValidationSet["ZNumber"]
    X2.set_index("ZNumber", inplace = True)
    
    Xnew = X2
    #Make predictions using the classifier model
    ynew = classifier.predict_classes(Xnew)
    
    #assign the predictions to the students
    SecondValidationSet["IsThereAProblem?"] = ynew
    
    SecondValidationSet.to_csv("AI Predictions2.csv")

    
if __name__ == "__main__":
    main()





    

    
