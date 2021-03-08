.PHONY: create_environment register_ipykernel orthophoto tiles train_test_split \
	download_zenodo_split response_tiles train_classifiers classify_tiles \
	tree_canopy confusion_df

#################################################################################
# GLOBALS                                                                       #
#################################################################################

## variables
PROJECT_NAME = jonction-tree-canopy

DATA_DIR = data
DATA_RAW_DIR := $(DATA_DIR)/raw
DATA_INTERIM_DIR := $(DATA_DIR)/interim
DATA_PROCESSED_DIR := $(DATA_DIR)/processed

MODELS_DIR = models

CODE_DIR = jonction_tree_canopy

## rules
define MAKE_DATA_SUB_DIR
$(DATA_SUB_DIR): | $(DATA_DIR)
	mkdir $$@
endef
$(DATA_DIR):
	mkdir $@
$(foreach DATA_SUB_DIR, \
	$(DATA_RAW_DIR) $(DATA_INTERIM_DIR) $(DATA_PROCESSED_DIR), \
	$(eval $(MAKE_DATA_SUB_DIR)))
$(MODELS_DIR):
	mkdir $@


#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up python interpreter environment
create_environment:
	conda env create -f environment.yml

## Register the environment as an IPython kernel for Jupyter
register_ipykernel:
	python -m ipykernel install --user --name $(PROJECT_NAME) \
		--display-name "Python ($(PROJECT_NAME))"

## 1. Download and unzip the orthoimage
### variables
ORTHOPHOTO_DIR := $(DATA_RAW_DIR)/orthophoto
ORTHOPHOTO_ZENODO_URI = \
	https://zenodo.org/record/4589559/files/orthophoto-2019.zip?download=1
ORTHOPHOTO_JPG := $(ORTHOPHOTO_DIR)/ORTHOPHOTOS_2019_40cm.jpg

### rules
$(ORTHOPHOTO_DIR): | $(DATA_RAW_DIR)
	mkdir $@
$(ORTHOPHOTO_DIR)/%.zip: | $(ORTHOPHOTO_DIR)
	wget --no-use-server-timestamps $(ORTHOPHOTO_ZENODO_URI) -O $@
$(ORTHOPHOTO_DIR)/%.jpg: $(ORTHOPHOTO_DIR)/%.zip
	unzip $< -d $(ORTHOPHOTO_DIR)
	touch $@
orthophoto: $(ORTHOPHOTO_JPG)

## 2. Split it into tiles
### variables
REF_RASTER_ZENODO_URI = \
	https://zenodo.org/record/4589559/files/jonctioncarte.tif?download=1
REF_RASTER_TIF := $(DATA_RAW_DIR)/jonctioncarte.tif
TILES_DIR := $(DATA_INTERIM_DIR)/tiles
TILES_CSV := $(TILES_DIR)/tiles.csv
#### code
MAKE_TILES_PY := $(CODE_DIR)/make_tiles.py

### rules
$(REF_RASTER_TIF): | $(DATA_RAW_DIR)
	wget --no-use-server-timestamps $(REF_RASTER_ZENODO_URI) -O $@
$(TILES_DIR): | $(DATA_INTERIM_DIR)
	mkdir $@
$(TILES_CSV): $(ORTHOPHOTO_JPG) $(REF_RASTER_TIF) $(MAKE_TILES_PY) | $(TILES_DIR)
	python $(MAKE_TILES_PY) $(ORTHOPHOTO_JPG) $(REF_RASTER_TIF) \
		$(TILES_DIR) $@
tiles: $(TILES_CSV)

## 3. Compute the train/test split
### variables
SPLIT_CSV_ZENODO_URI = \
	https://zenodo.org/record/4589559/files/split.csv?download=1
SPLIT_CSV := $(TILES_DIR)/split.csv
NUM_COMPONENTS = 24
NUM_TILE_CLUSTERS = 2

### rules
$(SPLIT_CSV): $(TILES_CSV)
	detectree train-test-split --img-dir $(TILES_DIR) \
		--output-filepath $(SPLIT_CSV) \
		--num-components $(NUM_COMPONENTS) \
		--num-img-clusters $(NUM_TILE_CLUSTERS)
train_test_split: $(SPLIT_CSV)
download_zenodo_split: $(TILES_CSV)
	wget --no-use-server-timestamps $(SPLIT_CSV_ZENODO_URI) -O \
		$(SPLIT_CSV)

## 4. Make the response tiles
### variables
LIDAR_DIR := $(DATA_RAW_DIR)/lidar
LIDAR_TILES_ZENODO_URI = \
	https://zenodo.org/record/4589559/files/lidar-tiles.zip?download=1
LIDAR_TILES_ZIP := $(LIDAR_DIR)/lidar-tiles.zip
LIDAR_TILES_CSV := $(LIDAR_DIR)/lidar-tiles.csv
RESPONSE_TILES_DIR := $(DATA_INTERIM_DIR)/response-tiles
RESPONSE_TILES_CSV := $(RESPONSE_TILES_DIR)/response-tiles.csv
#### code
MAKE_RESPONSE_TILES_PY := $(CODE_DIR)/make_response_tiles.py

### rules
$(LIDAR_DIR): | $(DATA_RAW_DIR)
	mkdir $@
$(LIDAR_TILES_CSV): | $(LIDAR_DIR)
	wget $(LIDAR_TILES_ZENODO_URI) -O $(LIDAR_TILES_ZIP)
	unzip $(LIDAR_TILES_ZIP) -d $(LIDAR_DIR)
	rm $(LIDAR_TILES_ZIP)
	touch $@
$(RESPONSE_TILES_DIR): | $(DATA_INTERIM_DIR)
	mkdir $@
$(RESPONSE_TILES_CSV): $(SPLIT_CSV) $(LIDAR_TILES_CSV) \
	$(MAKE_RESPONSE_TILES_PY) | $(RESPONSE_TILES_DIR)
	python $(MAKE_RESPONSE_TILES_PY) $(SPLIT_CSV) $(LIDAR_DIR) \
		$(RESPONSE_TILES_DIR) $@
response_tiles: $(RESPONSE_TILES_CSV)

## 5. Train a classifier for each tile cluster
### variables
MODEL_JOBLIB_FILEPATHS := $(foreach CLUSTER_LABEL, \
	$(shell seq 0 $$(($(NUM_TILE_CLUSTERS)-1))), \
	$(MODELS_DIR)/$(CLUSTER_LABEL).joblib)

### rules
$(MODELS_DIR)/%.joblib: | $(MODELS_DIR)
	detectree train-classifier --split-filepath $(SPLIT_CSV) \
		--response-img-dir $(RESPONSE_TILES_DIR) --img-cluster $* \
		--output-filepath $@
train_classifiers: $(MODEL_JOBLIB_FILEPATHS)

## 6. Classify the tiles
### variables
REFINE_BETA = 30
CLASSIFIED_TILES_DIR := $(DATA_INTERIM_DIR)/classified-tiles
CLASSIFIED_TILES_CSV_FILEPATHS := $(foreach CLUSTER_LABEL, \
	$(shell seq 0 $$(($(NUM_TILE_CLUSTERS)-1))), \
	$(CLASSIFIED_TILES_DIR)/$(CLUSTER_LABEL).csv)

### rules
$(CLASSIFIED_TILES_DIR): | $(DATA_INTERIM_DIR)
	mkdir $@
$(CLASSIFIED_TILES_DIR)/%.csv: $(MODELS_DIR)/%.joblib $(SPLIT_CSV) \
	$(MAKE_CLASSIFIED_TILES_PY) | $(CLASSIFIED_TILES_DIR)
	detectree classify-imgs $(SPLIT_CSV) --clf-filepath $< \
		--img-cluster $* --refine --refine-beta $(REFINE_BETA) \
		--output-dir $(CLASSIFIED_TILES_DIR)
	touch $@
classify_tiles: $(CLASSIFIED_TILES_CSV_FILEPATHS)

## 7. Mosaic the classified and response tiles into a single file
### variables
TREE_CANOPY_TIF := $(DATA_PROCESSED_DIR)/tree-canopy.tif
TREE_NODATA = 0  # shouldn't be ugly hardcoded like that...

### rules
$(TREE_CANOPY_TIF): $(RESPONSE_TILES_CSV) $(CLASSIFIED_TILES_CSV_FILEPATHS) \
	| $(DATA_PROCESSED_DIR)
	gdal_merge.py -o $@ -a_nodata $(TREE_NODATA) \
		$(wildcard $(CLASSIFIED_TILES_DIR)/*.tif) \
		$(wildcard $(RESPONSE_TILES_DIR)/*.tif)
tree_canopy: $(TREE_CANOPY_TIF)

## 8. Validation - confusion data frame
### variables
#### note: `VALIDATION_TILE_TIF` has been randomly sampled from `SPLIT_CSV`.
####       we hardcode it to the Makefile (ugly, I know) because the access to
####       LIDAR data at SITG has to be done manually at
####       https://www.etat.ge.ch/geoportail/pro/?method=showextractpanel
VALIDATION_TILE_TIF := $(TILES_DIR)/tile_4608-1536.tif
VALIDATION_TILES_DIR := $(DATA_INTERIM_DIR)/validation-tiles
CONFUSION_CSV := $(DATA_PROCESSED_DIR)/confusion.csv
#### code
MAKE_CONFUSION_DF_PY := $(CODE_DIR)/make_confusion_df.py

### rules
$(VALIDATION_TILES_DIR): | $(DATA_INTERIM_DIR)
	mkdir $@
$(CONFUSION_CSV): $(VALIDATION_TILE_TIF) $(SPLIT_CSV) $(MODEL_JOBLIB_FILEPATHS) \
	| $(VALIDATION_TILES_DIR) $(DATA_PROCESSED_DIR)
	python $(MAKE_CONFUSION_DF_PY) $(VALIDATION_TILE_TIF) $(SPLIT_CSV) \
		$(MODELS_DIR) $(LIDAR_DIR) $(VALIDATION_TILES_DIR) $@
confusion_df: $(CONFUSION_CSV)



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := show-help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: show-help
show-help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) == Darwin && echo '--no-init --raw-control-chars')
