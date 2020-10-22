# How to evaluate stuff
pip install sentence-transformers -U
pip install easy-table
dokumente aus evaluation/attacked_documents cleanen (use utility.read_labeled_data). Pro clean methode einen eigenen ordner in evaluation/cleaned erstellen und die gecleante datei mit selben namen wie die attackte datei darin speichern.
das Ende von eval_documents.py manipulieren, so dass in documents die paths zu deinen gecleanten dateien stehen.
Ausführen und über results in evaluation.csv freuen.
make_tables.py ausführen und über tables freuen.