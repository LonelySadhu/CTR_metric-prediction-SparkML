 - PySparkJob.py - обработка данных и подготовка их к обучению моделей.  
Для этого выделим ключевые параметры и определим целевую фичу CTR (click-through rate).     
При этом наши данные могут впоследствии изменяться, накапливаться и достигать больших объемов,  
поэтому необходимо реализовать задачу обработки данных, которую при необходимости сможем многократно выполнять для получения результата над любым объемом данных.    
Структура данных:  

date - день, в который происходят события 
time - точное время события 
event -	тип события, может быть или показ или клик по рекламе 
platform -	платформа, на которой произошло рекламное событие 
ad_id -	id рекламного объявления 
client_union_id -	id рекламного клиента 
campaign_union_id -	id рекламной кампании 
ad_cost_type - тип объявления с оплатой за клики (CPC) или за показы (CPM) 
ad_cost -	стоимость объявления в рублях, для CPC объявлений - это цена за клик, для CPM - цена за 1000 показов 
has_video -	есть ли у рекламного объявления видео 
target_audience_count -	размер аудитории, на которую таргетируется объявление 

- PySparkMLFit.py - задача, которая должна тренировать модель, подбирать оптимальные гиперпараметры на входящих данных, сохранять ее и производить оценку качества модели,   используя RegressionEvaluator и выводя в консоль RMSE модели на основе test датасета.  
Варианты запуска задачи:  

spark-submit PySparkMLFit.py train.parquet test.parquet  
#или  
python PySparkMLFit.py train.parquet test.parquet  
где:  
train.parquet - путь к датасету, который необходимо использовать для обучения  
test.parquet - путь к датасету, который необходимо использовать для оценки полученной модели  

- PySparkMLPredict.py - задача, которая должна загружать модель и строить предсказание над переданными ей данными.  
Варианты запуска задачи: 
spark-submit PySparkMLPredict.py test.parquet result  
#или
python PySparkMLPredict.py test.parquet result  
где:  
test.parquet - путь к датасету, на основе данных которого нужно выполнить предсказания CTR  
result - путь, по которому будет сохранен результат предсказаний следующего вида [ad_id, prediction]    
