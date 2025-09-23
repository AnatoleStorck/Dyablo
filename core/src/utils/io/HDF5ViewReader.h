#pragma once

#include <hdf5.h>
#include <hdf5_hl.h>

#include "HDF5ViewWriter.h"

namespace dyablo{

class HDF5ViewReader {

public:
    /**
     * Open an existing hdf5 file at path `filename`
     */
    HDF5ViewReader(const std::string& filename)
    {
        m_hdf5_file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    }

    ~HDF5ViewReader()
    {
        close();
    }

    void close()
    {
        if (m_hdf5_file) {
            H5Fclose(m_hdf5_file);
            m_hdf5_file = 0;
        }
    }

    template<typename View_t,
            typename = std::enable_if_t<std::is_same_v<typename View_t::array_layout, Kokkos::LayoutRight>>>
    View_t read_dataset(const std::string& varpath)
    {
        hid_t hdf5_type = hdf5_type_id<typename View_t::value_type>();
        constexpr hid_t rank = (hid_t)View_t::rank;

        hid_t dataset_properties = H5Pcreate(H5P_DATASET_ACCESS);
        hid_t dataset = H5Dopen2(m_hdf5_file, varpath.c_str(), dataset_properties );
        DYABLO_ASSERT_HOST_RELEASE(dataset >= 0, "Failed to open dataset " << varpath);

        hid_t filespace = H5Dget_space(dataset);
        DYABLO_ASSERT_HOST_RELEASE( rank == H5Sget_simple_extent_ndims(filespace), "hdf5 error : dataset rank doesn't match view rank" );

        hsize_t dims[rank], maxdims[rank];
        H5Sget_simple_extent_dims( filespace, dims, maxdims );

        View_t data_d;
        if constexpr (rank == 1)
            data_d = View_t(varpath, dims[0]);
        else if constexpr (rank == 2)
            data_d = View_t(varpath, dims[0], dims[1]);
        else if constexpr (rank == 3)
            data_d = View_t(varpath, dims[0], dims[1], dims[2]);

        // Allocate Kokkos view
        {
            hid_t read_properties = H5Pcreate(H5P_DATASET_XFER);
            #ifdef HDF5_IS_CUDA_AWARE
                H5Dread( dataset, hdf5_type, filespace, filespace, read_properties, data_d.data() );
            #else
                auto data_h = Kokkos::create_mirror_view( data_d );
                H5Dread( dataset, hdf5_type, filespace, filespace, read_properties, data_h.data() );
                Kokkos::deep_copy( data_d, data_h );
            #endif
            H5Pclose(read_properties);
        }

        H5Dclose(dataset);
        H5Sclose(filespace);
        H5Pclose(dataset_properties);

        return data_d;
    }

    template< typename View_t >
    View_t read_attr( const std::string& varpath, const std::string& attrname )
    {
        hsize_t dims[View_t::rank];
        hid_t type_id = hdf5_type_id<typename View_t::value_type>();

        hid_t dataset = H5Dopen2(m_hdf5_file, varpath.c_str(), H5P_DEFAULT);

        hid_t attr = H5Aopen(dataset, attrname.c_str(), H5P_DEFAULT);
        hid_t aspace = H5Aget_space(attr);
        H5Sget_simple_extent_dims(aspace, dims, NULL);

        View_t data_d;
        if constexpr (View_t::rank == 1)
            data_d = View_t(attrname, dims[0]);
        else if constexpr (View_t::rank == 2)
            data_d = View_t(attrname, dims[0], dims[1]);
        else if constexpr (View_t::rank == 3)
            data_d = View_t(attrname, dims[0], dims[1], dims[2]);

        herr_t status;
    #ifdef HDF5_IS_CUDA_AWARE
        status = H5Aread(attr, type_id, data_d.data());
    #else
        {
            auto data_h = Kokkos::create_mirror_view(data_d);
            status = H5Aread(attr, type_id, data_h.data());
            Kokkos::deep_copy(data_d, data_h);
        }
    #endif
        DYABLO_ASSERT_HOST_RELEASE( status == 0, "hdf5 error : reading attribute " << attrname << " from " << varpath );
        H5Sclose(aspace);
        H5Aclose(attr);
        H5Dclose(dataset);

        return data_d;
    }

private:
  hid_t m_hdf5_file; // HDF5 file descriptor
};

}  // namespace dyablo