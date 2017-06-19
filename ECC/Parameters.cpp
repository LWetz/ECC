#include "atf_library/atf.h"

void tuneClassify() {
	int ensembleSize = 10;
	int forestSize = 10;
	int dataSize = 10;
	auto tp_NUM_WG_CHAINS_SC = atf::tp("NUM_WG_CHAINS_SC", atf::interval(1,ensembleSize),
									   [&](auto tp_NUM_WG_CHAINS_SC){ return ensembleSize % tp_NUM_WG_CHAINS_SC == 0; });
	auto tp_NUM_WG_INSTANCES_SC = atf::tp("NUM_WG_INSTANCES_SC", atf::interval(1,dataSize),
										  [&](auto tp_NUM_WG_INSTANCES_SC){ return dataSize % tp_NUM_WG_INSTANCES_SC == 0; });
	auto tp_NUM_WG_TREES_SC = atf::tp("NUM_WG_TREES_SC", atf::interval(1,forestSize),
									  [&](auto tp_NUM_WG_TREES_SC){ return forestSize % tp_NUM_WG_TREES_SC == 0; });
	auto tp_NUM_WI_CHAINS_SC = atf::tp("NUM_WI_CHAINS_SC", atf::interval(1,ensembleSize),
									   [&](auto tp_NUM_WI_CHAINS_SC){ return ensembleSize / tp_NUM_WG_CHAINS_SC % tp_NUM_WI_CHAINS_SC == 0; });
	auto tp_NUM_WI_INSTANCES_SC = atf::tp("NUM_WI_INSTANCES_SC", atf::interval(1, dataSize),
										  [&](auto tp_NUM_WI_INSTANCES_SC){ return dataSize / tp_NUM_WG_INSTANCES_SC % tp_NUM_WI_INSTANCES_SC == 0; });
	auto tp_NUM_WI_TREES_SC = atf::tp("NUM_WI_TREES_SC", atf::interval(1,forestSize),
									  [&](auto tp_NUM_WI_TREES_SC){ return forestSize / tp_NUM_WG_TREES_SC % tp_NUM_WI_TREES_SC == 0; });

	// TODO buffer erstellen
	std::string source;
	std::vector<int> a, b, int_res, intermediateBuffer;

	auto kernel_1 = ocl_md_hom({"NVIDIA", atf::cf::device_info::GPU, 0},
							   {source, "gemv_1"},
							   atf::inputs(atf::buffer(a), atf::buffer(b), atf::buffer(int_res), atf::buffer(intermediateBuffer)), //intermediate letzter und nur I*C
							   atf::cf::GS(tp_NUM_WG_INSTANCES_SC * tp_NUM_WI_INSTANCES_SC, tp_NUM_WG_CHAINS_SC * tp_NUM_WI_CHAINS_SC, tp_NUM_WG_TREES_SC * tp_NUM_WI_TREES_SC),
							   atf::cf::LS(tp_NUM_WI_INSTANCES_SC                         , tp_NUM_WI_CHAINS_SC,                       tp_NUM_WI_TREES_SC),
							   {tp_NUM_WG_TREES_SC.name()});

//	auto tuner = atf::exhaustive();
	auto tuner = atf::open_tuner(atf::cond::evaluations(1000));

	auto best_config = tuner(
			G(tp_NUM_WG_CHAINS_SC, tp_NUM_WI_CHAINS_SC),
			G(tp_NUM_WG_INSTANCES_SC, tp_NUM_WI_INSTANCES_SC),
			G(tp_NUM_WG_TREES_SC, tp_NUM_WI_TREES_SC)
	)(kernel_1);





	const int best_NUM_WG_TREES_SC = best_config["NUM_WG_TREES_SC"].value();

		auto tp_NUM_WG_CHAINS_SR = atf::tp("NUM_WG_CHAINS_SR", atf::interval(1,ensembleSize),
										   [&](auto tp_NUM_WG_CHAINS_SR){ return ensembleSize % tp_NUM_WG_CHAINS_SR == 0; });
		auto tp_NUM_WG_INSTANCES_SR = atf::tp("NUM_WG_INSTANCES_SR", atf::interval(1,dataSize),
											  [&](auto tp_NUM_WG_INSTANCES_SR){ return dataSize % tp_NUM_WG_INSTANCES_SR == 0; });
		auto tp_NUM_WI_CHAINS_SR = atf::tp("NUM_WI_CHAINS_SR", atf::interval(1,ensembleSize),
										   [&](auto tp_NUM_WI_CHAINS_SR){ return ensembleSize / tp_NUM_WG_CHAINS_SR % tp_NUM_WI_CHAINS_SR == 0; });
		auto tp_NUM_WI_TREES_SR = atf::tp("NUM_WI_TREES_SR", atf::interval(1,best_NUM_WG_TREES_SC),
										  [&](auto tp_NUM_WI_TREES_SR){ return best_NUM_WG_TREES_SC % tp_NUM_WI_TREES_SR == 0; });
		auto tp_NUM_WI_INSTANCES_SR = atf::tp("NUM_WI_INSTANCES_SR", atf::interval(1, dataSize),
											  [&](auto tp_NUM_WI_INSTANCES_SR){ return dataSize / tp_NUM_WG_INSTANCES_SR % tp_NUM_WI_INSTANCES_SR == 0; });


//		auto tp_NUM_WG_LABELS_FC = atf::tp("NUM_WG_LABELS_FC", atf::interval(0,100));
//		auto tp_NUM_WG_CHAINS_FC = atf::tp("NUM_WG_CHAINS_FC", atf::interval(0,100));
//		auto tp_NUM_WG_INSTANCES_FC = atf::tp("NUM_WG_INSTANCES_FC", atf::interval(0,100));
//		auto tp_NUM_WG_LABELS_FR = atf::tp("NUM_WG_LABELS_FR", atf::interval(0,100));
//		auto tp_NUM_WG_INSTANCES_FR = atf::tp("NUM_WG_INSTANCES_FR", atf::interval(0,100));
//
//		auto tp_NUM_WI_LABELS_FC = atf::tp("NUM_WI_LABELS_FC", atf::interval(0,100));
//		auto tp_NUM_WI_INSTANCES_FC = atf::tp("NUM_WI_INSTANCES_FC", atf::interval(0,100));
//		auto tp_NUM_WI_CHAINS_FC = atf::tp("NUM_WI_CHAINS_FC", atf::interval(0,100));
//		auto tp_NUM_WI_LABELS_FR = atf::tp("NUM_WI_LABELS_FR", atf::interval(0,100));
//		auto tp_NUM_WI_INSTANCES_FR = atf::tp("NUM_WI_INSTANCES_FR", atf::interval(0,100));
//		auto tp_NUM_WI_CHAINS_FR = atf::tp("NUM_WI_CHAINS_FR", atf::interval(0,100));
}

int main() {
	
	return 0;
}
