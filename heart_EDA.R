library(dplyr)
library(ggplot2)
library(forcats)
library(rsample)
library(tidyverse)
library(tidymodels)
library(gridExtra)
library(pROC)
library(tidyverse)
library(caret)
library(nnet)


cleveland <- read.csv(file = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", header = FALSE)
head(cleveland)

names = c("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "heart_disease")
colnames(cleveland) <- names

# cleveland <- cleveland %>%
#   mutate(heart_disease = case_when(heart_disease == 0 ~ 0,
#                                    (heart_disease > 0 ~ 1)))

# cleveland <- cleveland %>%
#   mutate(sex = case_when(sex == 0 ~ "female",
#                          sex == 1 ~ "male")) %>%
#   mutate(cp = case_when(cp == 1 ~ "typical angina",
#                         cp == 2 ~ "atypical angina",
#                         cp == 3 ~ "non-anginal pain",
#                         cp == 4 ~ "asymptomatic")) %>%
#   mutate(fbs = case_when(fbs == 1 ~ "high",
#                          fbs == 0 ~ "low")) %>%
#   mutate(exang = case_when(exang == 0 ~ "no",
#                            exang == 1 ~ "yes"))

cleveland$ca <- as.integer(cleveland$ca)
cleveland$thal <- as.integer(cleveland$thal)
cleveland <- cleveland %>% drop_na()

str(cleveland)


cleveland <- cleveland %>%
  mutate(sex = as.factor(sex)) %>%
  mutate(cp = as.factor(cp)) %>%
  mutate(fbs = as.factor(fbs)) %>%
  #     mutate(thalach = as.factor(thalach)) %>%
  mutate(restecg = as.factor(restecg)) %>%
  mutate(exang = as.factor(exang)) %>%
  mutate(slope = as.factor(slope)) %>%
  mutate(thal = as.factor(thal)) %>%
  mutate(heart_disease = as.factor(heart_disease))

Hearts <- cleveland


Hearts_cat <- Hearts %>%
    mutate(sex = case_when(sex == 0 ~ "female",
                           sex == 1 ~ "male")) %>%
    mutate(cp = case_when(cp == 1 ~ "typical angina",
                          cp == 2 ~ "atypical angina",
                          cp == 3 ~ "non-anginal pain",
                          cp == 4 ~ "asymptomatic")) %>%
    mutate(fbs = case_when(fbs == 1 ~ "high (>120)",
                           fbs == 0 ~ "low (<=120)")) %>%
    mutate(exang = case_when(exang == 0 ~ "no",
                             exang == 1 ~ "yes"))%>%
  mutate(restecg = case_when(restecg == 0 ~ "Normal",
                             restecg == 1 ~ "Abnormal",
                             restecg == 2 ~ " Left ventricular hypertrophy"))%>%
  mutate(thal = case_when(thal == 3 ~ "Normal",
                          thal == 6 ~ "Fixed defect",
                          thal == 7 ~ "Reversible defect"))%>%
  mutate(slope = case_when(slope == 1 ~ "Upsloping",
                           slope == 2 ~ "Flat",
                           slope == 3 ~ "Downsloping"))

par(mfrow=c(1,1))
col.heart_disease <- c("pink","blue")
plot(table(Hearts$sex,Hearts$heart_disease),xlab="Gender",ylab="Diagnostics",col=col.heart_disease, main=" ")


summary(Hearts)

ggplot(Hearts_cat, mapping = aes(x = heart_disease, fill = sex)) +
  geom_bar(position = "fill") +  scale_fill_brewer(palette = "Pastel1") +
  labs(
    title = "Heart Disease Proportion by Gender",x = "Heart disease severity", y = "Proportion"
 ) +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))


chisq.test(Hearts$heart_disease, Hearts$age, correct=FALSE)#p-value = 0.2986
chisq.test(Hearts$heart_disease, Hearts$sex, correct=FALSE)#p-value = 0.0001134
chisq.test(Hearts$heart_disease, Hearts$cp, correct=FALSE)#p-value = 5.522e-13
chisq.test(Hearts$heart_disease, Hearts$trestbps, correct=FALSE)#p-value = 0.1212
chisq.test(Hearts$heart_disease, Hearts$chol, correct=FALSE)#p-value = 0.08893
chisq.test(Hearts$heart_disease, Hearts$fbs, correct=FALSE)#p-value = 0.09433
chisq.test(Hearts$heart_disease, Hearts$restecg, correct=FALSE)#p-value = 0.0117
#chisq.test(Hearts$heart_disease, Hearts$thalch, correct=FALSE)
chisq.test(Hearts$heart_disease, Hearts$exang, correct=FALSE)#p-value = 7.299e-12
chisq.test(Hearts$heart_disease, Hearts$oldpeak, correct=FALSE)#p-value = 5.216e-10
chisq.test(Hearts$heart_disease, Hearts$slope, correct=FALSE)#p-value = 7.242e-09
chisq.test(Hearts$heart_disease, Hearts$ca, correct=FALSE)#p-value < 2.2e-16
chisq.test(Hearts$heart_disease, Hearts$thal, correct=FALSE)#p-value < 2.2e-16


#insignificant age,trestbps,chol,fbs
#significant sex+cp+restecg+thalch+exang+oldpeak+slope+ca+thal

library(corrplot)
heart_matrix <- data.matrix(Hearts, rownames.force = NA)
M <- cor(heart_matrix)
corrplot(M, method = "number", number.cex = 0.70, order="hclust")




ggplot(gather(Hearts[,sapply(Hearts, is.numeric)], cols, value), 
       aes(x = value)) + 
  geom_histogram(aes(fill = cols)) + 
  geom_density(aes(y = stat(count))) +
  facet_wrap(cols~., scales = "free") +
  theme(legend.position =  "none",
        axis.title.y = element_text(angle = 90))


# Hearts %>%
#   select(sex, age, trestbps, chol, heart_disease) %>%
#   pivot_longer(cols = c(age, trestbps, chol), names_to = "variable", values_to = "value") %>%
#   ggplot(aes(x = value, fill = variable)) +
#   geom_density(alpha = 0.5) +
#   facet_grid(heart_disease ~ sex, scales = "free_y") +
#   theme(legend.position = "bottom") +
#   labs(x = "Value", y = "Density", fill = "Variable")

library(ggplot2)

# Count the values in the target column
target_counts <- table(Hearts$heart_disease)

# Create a dataframe from the counts
df_counts <- data.frame(heart_disease = names(target_counts),
                        count = as.numeric(target_counts))

# Plot the counts as a bar plot
ggplot(df_counts, aes(x = heart_disease, y = count, fill = heart_disease)) +
  geom_bar(stat = "identity", width = 0.1) +
  #scale_fill_manual(values = c("salmon", "lightgreen","blue","green","pink")) +
  labs(x = "Presence of heart disease",
       y = "No. of people",
       title = "Counts of heart disease presence/absence") +
  theme_bw() +
  theme(legend.position = "bottom",
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 12),
        legend.title = element_blank(),
        legend.text = element_text(size = 12))

library(ggplot2)
library(dplyr)
library(readr)
library(qgraph)
library(caret)
library(tidyverse)



library(ggplot2)
###good
ggplot(Hearts, aes(x = sex, fill = factor(heart_disease))) +
  geom_bar(position = "dodge") + 
  scale_fill_brewer(palette = "Pastel2") +
  #scale_fill_manual(values = c("#3C5488", "#F29F05", "#EFE0CE", "#4B4B4B", "#BDBDBD")) +
  labs(x = "Gender", y = "Count", fill = "Heart Disease") +
  ggtitle("Heart Disease Presence/Absence by Gender") +
  theme_bw()
####

ggplot(Hearts, aes(x = age, fill = factor(heart_disease))) +
  geom_histogram(alpha = 0.5, position = "identity", bins = 15) +
  #scale_fill_brewer(palette = "Pastel2") +
  scale_fill_manual(values = c("#3C5488", "#F29F05", "#EFE0CE", "#4B4B4B", "#BDBDBD")) +
  labs(x = "Age", y = "Count", fill = "Heart Disease") +
  ggtitle("Age Distribution by Heart Disease Presence/Absence") +
  theme_bw()

ggplot(Hearts, aes(x = factor(heart_disease), y = trestbps, fill = factor(heart_disease))) +
  geom_boxplot() +scale_fill_brewer(palette = "Pastel1") +
 # scale_fill_manual(values = c("#3C5488", "#F29F05", "#EFE0CE", "#4B4B4B", "#BDBDBD")) +
  labs(x = "Heart Disease", y = "Resting Blood Pressure", fill = "Heart Disease") +
  ggtitle("Resting Blood Pressure by Heart Disease Presence/Absence") +
  theme_bw()

ggplot(Hearts, aes(x = heart_disease, y = age, fill = factor(heart_disease))) +
  geom_boxplot() +
  scale_fill_brewer(palette = "Pastel1") +
  labs(title = "Distribution of age by heart disease status", 
       x = "Heart disease status", y = "Age",fill = "Heart Disease") +
  theme_bw()

ggplot(Hearts, aes(x = age, y = thalach, color = factor(heart_disease))) +
  geom_point() +
  labs(title = "Relationship between age and \n maximum heart rate achieved, \n colored by heart disease status", 
       x = "Age", y = "Maximum heart rate achieved during exercise \n (thalach)" ,fill = "Heart Disease Severity") +
  theme_bw()

cleveland1 <- cleveland %>%
  mutate(sex = case_when(sex == 0 ~ "female",
                         sex == 1 ~ "male")) %>%
  mutate(cp = case_when(cp == 1 ~ "typical angina",
                        cp == 2 ~ "atypical angina",
                        cp == 3 ~ "non-anginal pain",
                        cp == 4 ~ "asymptomatic")) %>%
  mutate(fbs = case_when(fbs == 1 ~ "high",
                         fbs == 0 ~ "low")) %>%
  mutate(exang = case_when(exang == 0 ~ "no",
                           exang == 1 ~ "yes"))%>%
         mutate(thal = case_when(thal == 3 ~ "Normal",
                                  thal == 6 ~ "Fixed Defect",
                                  thal == 7 ~ "Reversible Defect"))

ggplot(cleveland1, aes(x = factor(cp), fill = factor(heart_disease))) +
  geom_bar(position = "dodge") +
  scale_fill_brewer(palette = "Pastel1") +
  labs(title = "Frequency of chest pain type by heart disease status \n (cp)", 
       x = "Chest pain type", y = "Count") +
  theme_bw()


ggplot(cleveland1, aes(x = factor(cp), fill = factor(heart_disease))) +
  geom_bar(position = "dodge") +
  scale_fill_brewer(palette = "Pastel1") +
  labs(title = "Frequency of chest pain type by heart disease status", 
       x = "Chest pain type", y = "Patient Count",fill = "Heart Disease") +
  theme_bw()

ggplot(cleveland1, aes(x = factor(thal), fill = factor(heart_disease))) +
  geom_bar(position = "stack") +
  scale_fill_brewer(palette = "Pastel1") +
  labs(title = "Frequency of thalassemia type by heart disease status", 
       x = "Thalassemia type", y = "Patient Count",fill = "Heart Disease Severity") +
  theme_bw()

library(ggplot2)

# Load the Cleaveland heart database
data(Hearts)

# Filter for relevant columns
df <- Hearts[c("age", "sex", "fbs", "chol", "heart_disease")]

# Convert sex to factor with descriptive labels
df$sex <- factor(df$sex, levels = c(0, 1), labels = c("Female", "Male"))

# Convert num to factor with descriptive labels
df$heart_disease <- factor(df$heart_disease, levels = c(0, 1, 2, 3, 4),
                 labels = c("No disease", "Low Severity", "Moderate Severity", "High Severity", "Very High Severity"))

# Create a scatterplot matrix with num as the color variable
ggplot(df, aes(x = age, y = chol, color = heart_disease)) +
  geom_point(size = 3, alpha = 0.7) +
  facet_grid(sex~fbs) +
  labs(x = "Age", y = "Cholesterol", color = "Heart disease",title = "categorization by fasting blood sugar") +
  scale_color_brewer(palette = "Set3") +
  theme_bw()


ggplot(Hearts, aes(x = trestbps, y = chol)) +
  geom_point() +
  labs(x = "Resting Blood Pressure",
       y = "Cholesterol",
       title = "Relationship between Resting Blood Pressure and Cholesterol") +
  theme_bw()


library(ggplot2)

# Load the Cleveland heart disease dataset
data(heart)

counts <- Hearts_cat %>%
  group_by(sex, heart_disease) %>%
  summarize(count = n())
# Count the number of patients by gender and heart disease status
patient_counts <- aggregate(Hearts_cat$age, by = list(gender = Hearts_cat$sex, heart_disease = Hearts_cat$heart_disease), length)
colnames(patient_counts) <- c("gender", "heart_disease", "num_patients")

# Create a stacked bar plot of patient counts
ggplot(patient_counts, aes(x = gender, y = num_patients, fill = factor(heart_disease))) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = num_patients), position = position_stack(vjust = 0.5)) +
  labs(x = "gender", y = "Number of patients", fill = "Heart disease status", title = "Patient count by Gender and disease severity")+ 
  #scale_fill_manual(values = c("#E69F00", "#56B4E9")) +
  theme_minimal()


