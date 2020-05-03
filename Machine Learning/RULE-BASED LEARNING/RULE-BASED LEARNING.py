import Orange
from sklearn.metrics import accuracy_score, precision_score , recall_score, f1_score

#Loading data
wine = Orange.data.Table("wine")

#Creating cross validation model and 3 learners
Cross_val = Orange.evaluation.testing.CrossValidation(store_models=True)

learner1 = Orange.classification.rules.CN2Learner()
learner.rule_finder.evaluator = Orange.classification.rules.EntropyEvaluator
learner.rule_finder.search_algorithm.beam_width = 5
learner.rule_finder.general_validator.max_rule_length = 5
learner.rule_finder.general_validator.min_covered_examples = 15

learner_unorder = Orange.classification.CN2UnorderedLearner()
learner_unorder.rule_finder.evaluator = Orange.classification.rules.LaplaceAccuracyEvaluator
learner_unorder.rule_finder.search_algorithm.beam_width = 5
learner_unorder.rule_finder.general_validator.max_rule_length = 5
learner_unorder.rule_finder.general_validator.min_covered_examples = 10

learner2 = Orange.classification.rules.CN2Learner()
learner2.rule_finder.evaluator = Orange.classification.rules.LaplaceAccuracyEvaluator
learner2.rule_finder.search_algorithm.beam_width = 5
learner2.rule_finder.general_validator.max_rule_length = 5
learner2.rule_finder.general_validator.min_covered_examples = 10


#Training learners 
learn = [learner1, learner_unorder , learner2]
RBL = Cross_val(learners= learn, data= wine )

#Printing the metrics and rules
print("For ordered CN2 entropy evaluator accuracy is: " , accuracy_score(RBL.actual, RBL.predicted[0]))
print ("Precision Score is:", precision_score(RBL.actual, RBL.predicted[0], average="macro") )
print ("Recall Score is:", recall_score(RBL.actual, RBL.predicted[0], average="macro") )
print ("F1 Score is:", f1_score(RBL.actual, RBL.predicted[0], average="macro") )
print()
for rule in RBL.models[0][0].rule_list:
    print(rule)


print("\nFor unordered CN2 laplace evaluator accuracy is: ",  accuracy_score(RBL.actual, RBL.predicted[1]))
print ("Precision Score is:", precision_score(RBL.actual, RBL.predicted[1], average="macro") )
print ("Recall Score is:", recall_score(RBL.actual, RBL.predicted[1], average="macro") )
print ("F1 Score is:", f1_score(RBL.actual, RBL.predicted[1], average="macro") )
print()
for rule in RBL.models[1][0].rule_list:
    print(rule)
    
    
print("\nFor ordered CN2 laplace evaluator accuracy is: " , accuracy_score(RBL.actual, RBL.predicted[2]))
print ("Precision Score is:", precision_score(RBL.actual, RBL.predicted[2], average="macro") )
print ("Recall Score is:", recall_score(RBL.actual, RBL.predicted[2], average="macro") )
print ("F1 Score is:", f1_score(RBL.actual, RBL.predicted[2], average="macro") )
print()
for rule in RBL.models[2][0].rule_list:
    print(rule)